#!/usr/bin/env python3
"""
CERN Dataset - Fixed-Point Quantization Script

Loads each trained FasterKAN model from cern_dataset/train/, applies
fixed-point quantization to the KAN submodule, and saves the quantized
model alongside the original checkpoint.

Usage:
    python quant_model.py [--train-dir TRAIN_DIR] [--bits BITS] [--no-fit]

The script discovers all training runs under TRAIN_DIR (default: ./train/),
quantizes each best checkpoint it finds, and saves the result as
<run_dir>/quantized_models/best/best_quantized.pt with the quantization config written
to <run_dir>quantized_models/quant_config.json.
"""
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Quantize trained FasterKAN models for the CERN dataset.')
    parser.add_argument('-d', '--train-dir', dest='train_dir', default=os.path.join(THIS_DIR, 'train'), help='Top-level training output directory (default: ./train)')
    parser.add_argument('--bits', dest='num_bits', type=int, default=16, help='Base bit-width for quantization (default: 16)')
    parser.add_argument('--data-channels', dest='data_channels', type=int, default=1, help='Parallel input channels (default: 1)')
    parser.add_argument('--rslt-channels', dest='rslt_channels', type=int, default=1, help='Outputs per iteration (default: 1)')
    parser.add_argument('--no-fit', dest='fit_model', action='store_false', default=True, help='Skip calibration-based frac_bits fitting')
    parser.add_argument('--hardtanh', dest='hardtanh', action='store_true', default=False, help='Use integer Hardtanh approximation (RSWAFF mode only)')
    parser.add_argument('--mixed', dest='mixed', action='store_true', default=False, help='Use the mixedDW representation')
    parser.add_argument('--signed-actf', dest='signed_actf', action='store_true', default=False, help='Use signed activation function')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False, help='Process all models in the train directory (default behaviour)')
    sel.add_argument('-m','--model', dest='model_name', type=str, default=None, help='Process a specific run by folder name, e.g. test_0 or <hash>/test_0')

    args = parser.parse_args()

    # ------------------------------------------------------------------ 
    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config.config import load_config, instantiate
    from kan_utils.utils import load_model, save_model, save_dict
    from kan_utils.quantization import FixedPointFasterKAN, FloatWrapperModule
    from kan_utils.quantization.old.parameter_transform import packetize_model_to_bin
    from prepare_dataset import build_dataset, get_dataset_paths, normalize_data
    from custom_dataset import DistributedH5Dataset


    # -------- Quantization configuration
    NUM_BITS_LOW  = args.num_bits // (1 + int(args.mixed))
    NUM_BITS_HIGH = args.num_bits 
    # NUM_BITS = 8

    QUANT_DTYPE = {
        'grid'   : (NUM_BITS_HIGH,  False),
        'scale'  : (NUM_BITS_HIGH,  True),
        'weight' : (NUM_BITS_LOW,   False),
        'sdff'   : (NUM_BITS_HIGH,  False),
        'actf'   : (NUM_BITS_HIGH, not args.signed_actf),
        'result' : (NUM_BITS_HIGH,  False),
    }
    QUANT_FRAC_BITS = {
        'grid'   : NUM_BITS_HIGH // 2,
        'scale'  : NUM_BITS_HIGH // 2,
        'weight' : NUM_BITS_LOW  // 2,
        'sdff'   :(NUM_BITS_HIGH) // 2,
        'actf'   : NUM_BITS_HIGH ,
        'result' : NUM_BITS_HIGH // 2,
    }

    device = torch.device('cpu')

    # ------ Collect all training runs from train dir
    train_dir = args.train_dir
    runs = []
    for hash_name in os.listdir(train_dir):
        hash_path = os.path.join(train_dir, hash_name)
        if not os.path.isdir(hash_path):
            continue
        for run_name in sorted(os.listdir(hash_path)):
            run_path        = os.path.join(hash_path, run_name)
            best_model_path = os.path.join(run_path, 'models', f'{args.epoch}.pt')
            model_cfg_path  = os.path.join(run_path, 'config', 'model.json')
            train_cfg_path  = os.path.join(run_path, 'config', 'train.json')

            if not os.path.exists(best_model_path):
                # print(f'[SKIP] No "{args.epoch}" model at {best_model_path}')
                continue
            runs.append({
                'run_path'        : run_path,
                'best_model_path' : best_model_path,
                'model_cfg_path'  : model_cfg_path,
                'train_cfg_path'  : train_cfg_path,
            })

    if not runs:
        print('No trained models found. Run train_model.py first.')
        sys.exit(0)

    # ------ Filter runs when --model is given
    if args.model_name is not None:
        def _matches(run_path, name):
            run_name  = os.path.basename(run_path)
            hash_name = os.path.basename(os.path.dirname(run_path))
            return name in (run_name, hash_name, f'{hash_name}/{run_name}')

        runs = [r for r in runs if _matches(r['run_path'], args.model_name)]
        if not runs:
            print(f'[ERROR] No run matching "{args.model_name}" found under {train_dir}')
            sys.exit(1)
        print(f'Selected {len(runs)} run(s) matching "{args.model_name}".')


    # ------ Build calibration DataLoader (shared across all runs)
    # Only needed when fit_model=True, skipped with --no-fit
    calib_data = None
    if args.fit_model:
        first_model_cfg = load_config(runs[0]['model_cfg_path'])
        first_train_cfg = load_config(runs[0]['train_cfg_path'])

        build_dataset() 
        calib_files, _ = get_dataset_paths(['train', 'val'], split=first_train_cfg['splits'])

        calib_dataset = DistributedH5Dataset(
            h5files               = calib_files,
            buffer_size           = 2048,
            input_cols            = first_model_cfg['input'],
            output_cols           = first_model_cfg['output'],
            remove_mass_pt_window = first_model_cfg['remove_mass_pt_window'],
            preprocess_data       = lambda data, features = first_model_cfg['input']: normalize_data(data, features),
            preprocess_targ       = None,
        )
        calib_loader = DataLoader(calib_dataset, batch_size=512)
        calib_data, _ = next(iter(calib_loader))
        calib_data = calib_data.to(device)


    # ------ Quantize each run
    for run in runs:
        run_path        = run['run_path']
        best_model_path = run['best_model_path']
        model_cfg_path  = run['model_cfg_path']

        print(f'\n---- Quantizing: {run_path},  Epoch: {args.epoch} ----')

        # Reconstruct and load the full floating-point model
        model_config = load_config(model_cfg_path)
        model_fp     = instantiate(model_config, 'model').to(device)
        state_dict   = torch.load(best_model_path, map_location=device)
        model_fp.load_state_dict(state_dict)
        model_fp.eval()

        # Quantization to the KAN submodule only 
        kan_fp = model_fp.kan

        # Build the fixed-point quantized KAN
        quant_kan = FixedPointFasterKAN(
            model          = kan_fp,
            dtype_dict     = QUANT_DTYPE,
            frac_bits_dict = QUANT_FRAC_BITS,
            hardtanh       = args.hardtanh,
        ).to(device)

        per_layer_fitted = []
        if args.fit_model:
            print('  > Fitting frac_bits from calibration data ...')
            quant_kan.fit_quantize(calib_data, kan_fp)

            # Collect per-layer frac_bits produced by fit, then apply the
            # per-key minimum across all layers as the global frac_bits.
            per_layer_fitted = [dict(layer.frac_bits_dict) for layer in quant_kan.layers]
            all_keys = {k for d in per_layer_fitted for k in d}
            min_frac_bits = {k: min(d[k] for d in per_layer_fitted if k in d) for k in all_keys}
            # print(f'  [DEBUG] Per-layer fitted frac_bits: {per_layer_fitted}')
            # print(f'  [DEBUG] Applying global min frac_bits: {min_frac_bits}')
            
            if 'grid' in min_frac_bits and 'result' in min_frac_bits:
                min_grid_res = min(min_frac_bits['grid'], min_frac_bits['result'])
                min_frac_bits['grid'] = min_grid_res
                min_frac_bits['result'] = min_grid_res

            # Update quantizer with the global min frac_bits before quantization
            QUANT_FRAC_BITS.update(min_frac_bits)
            # Update the frac_bits in each layer according to the new QUANT_FRAC_BITS
            for layer in quant_kan.layers:
                layer.update_frac_bits(min_frac_bits)
            quant_kan.quantize(kan_fp)
        else:
            quant_kan.quantize(kan_fp)

        # Save quantized model weights
        models_dir = os.path.join(
            run_path, 
            'quantized_models', 
            args.epoch,
            ('mixed' if args.mixed else 'int') + f'{args.num_bits}'
        )
        os.makedirs(models_dir, exist_ok=True)
        out_path = os.path.join(models_dir, f'{args.epoch}_quantized.pt')
        save_model(quant_kan, out_path, device)
        print(f'  >> Saved quantized model to: {out_path}')

        # Export quantized parameters to binary files for hardware deployment
        # print(quant_kan.state_dict())
        packetize_model_to_bin(
            quant_kan.state_dict(), 
            models_dir, 
            DATA_CHANNELS = args.data_channels,
            RSLT_CHANNELS = args.rslt_channels, 
        )
        print(f'  >> Saved bin files to: {models_dir}/extracted_params/ and {models_dir}/packetized_params/')

        # Save the quantization config alongside the model
        quant_cfg = {
            'num_bits_low'   : NUM_BITS_LOW,
            'num_bits_high'  : NUM_BITS_HIGH,
            'hardtanh'       : args.hardtanh,
            'fit_model'      : args.fit_model,
            'dtype_dict'     : {k: list(v) for k, v in QUANT_DTYPE.items()},
            'frac_bits_dict' : QUANT_FRAC_BITS,
            'per_layer_frac_bits_fitted': per_layer_fitted,
            # NOTE: Maybe not needed, because the frac_bits_dict explains the quant config for each layer
            'per_layer_frac_bits_final': [
                layer.frac_bits_dict for layer in quant_kan.layers
            ],
        }
        save_dict(quant_cfg, os.path.join(models_dir, 'quant_config'))
        print(f'  >> Saved quant config to: {os.path.join(models_dir, 'quant_config.json')}')

        # Quick validation: compare float and quantized outputs on calib data
        if calib_data is not None:
            print('  > Validating quantized model outputs on calibration data ...')
            with torch.no_grad():
                out_fp   = kan_fp(calib_data)
                out_q    = FloatWrapperModule(quant_kan)(calib_data)
                max_err  = ((out_fp - out_q)**2).max().sqrt().item()
                mean_err = ((out_fp - out_q)**2).mean().sqrt().item()
            print(f'  Max output error:  {max_err:.6f}')
            print(f'  Mean output error: {mean_err:.6f}')
        else:
            print('  (Skipping error validation, no calibration data with --no-fit)')
            
    print("\n")
