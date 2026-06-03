#!/usr/bin/env python3
"""
MNIST Dataset - Fixed-Point Quantization Script

Loads each trained FasterKAN model from mnist/train/, applies
fixed-point quantization to the KAN submodule, and saves the quantized
model alongside the original checkpoint.

Usage:
    python quant_model.py [--train-dir test_dir] [--bits BITS] [--no-fit]

The script discovers all training runs under test_dir (default: ./train/),
quantizes each best checkpoint it finds, and saves the result as
<run_dir>/quantized_models/<epoch>/<epoch>_quantized.pt with the quantization config written
to <run_dir>/quantized_models/<epoch>/quant_config.json.
"""
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.insert(0, TOP_DIR)

    parser = ArgumentParser(description='Quantize trained FasterKAN models for the MNIST dataset.')
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('--bits', dest='num_bits', type=int, default=16, help='Base bit-width for quantization (default: 16)')
    parser.add_argument('--data-channels', dest='data_channels', type=int, default=1, help='Parallel input channels (default: 1)')
    parser.add_argument('--rslt-channels', dest='rslt_channels', type=int, default=1, help='Outputs per iteration (default: 1)')
    parser.add_argument('--no-fit', dest='fit_model', action='store_false', default=True, help='Skip calibration-based frac_bits fitting')
    parser.add_argument('--hardtanh', dest='hardtanh', action='store_true', default=False, help='Use integer Hardtanh approximation (RSWAFF mode only)')
    parser.add_argument('--mixed', dest='mixed', action='store_true', default=False, help='Use the mixedDW representation')
    parser.add_argument('--signed-actf', dest='signed_actf', action='store_true', default=False, help='Use signed activation function')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')
    parser.add_argument('--stage', dest='stage', type=int, default=1)
    parser.add_argument('-w', '--warn', dest='warn', action='store_true', help='Show warning messages')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False, help='Process all models in the train directory (default behaviour)')

    group = sel.add_argument_group()
    group.add_argument('-m','--model', dest='model_hash', type=str, nargs='*', help='Process a specific model by hash')
    group.add_argument('-t','--train', dest='train_hash', type=str, nargs='*', help='Process a specific training by hash')
    group.add_argument('-v','--version', '--test-version', dest='test_version', type=str, nargs='*', help='Process a specific training by hash')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all-heads', dest='all_heads', action='store_true', default=False, help='Process all heads of a model (default behaviour)')
    sel.add_argument('--head', dest='head', type=str, nargs='*', help='Process a specific head of a model by folder name, e.g. test_0 or <hash>')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    from warnings import warn
    
    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config import *
    from kan_utils.models import fuse_faster_kan
    from kan_utils.utils import load_model, save_model, save_dict, to
    from kan_utils.quantization import FixedPointFasterKAN, FloatWrapperModule
    from kan_utils.quantization.parameter_transform import packetize_model_to_bin
    
    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    from custom_dataset import SkinCancerDataset, get_extra_transforms
    from build_model import get_train_config_path, get_model_config_path, build_model, get_training_subdir
    import custom_model, extract_statistics

    # -------- Quantization configuration
    NUM_BITS_LOW  = args.num_bits // (1 + int(args.mixed))
    NUM_BITS_HIGH = args.num_bits

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
    print("------ Quantization Script for Skin Cancer Dataset ------")
    test_dir = args.test_dir
    
    # Find model hashes
    model_versions = [
        (model_hash, os.path.splitext(test_version)[0]) 
            for model_hash in os.listdir(
                os.path.join(test_dir, 'model_config')
            )
            for test_version in os.listdir(
                os.path.join(test_dir, 'model_config', model_hash)
            )
    ]
    train_versions = [
        (train_hash, os.path.splitext(test_version)[0]) 
            for train_hash in os.listdir(
                os.path.join(test_dir, 'train_config')
            )
            for test_version in os.listdir(
                os.path.join(test_dir, 'train_config', train_hash)
            )
    ]
    runs = []

    for train_version in train_versions:
        for model_version in model_versions:
            if train_version[-1] != model_version[-1]:
                continue
            curr_run = (
                model_version[0],
                *train_version,
            )
            pth = get_training_subdir(
                training_stage  = args.stage,
                model_hash      = curr_run[0],
                train_hash      = curr_run[1],
                test_version    = curr_run[2],
                top_dir         = test_dir
            )
            pth = os.path.join(
                get_training_subdir(
                    training_stage  = args.stage,
                    model_hash      = curr_run[0],
                    train_hash      = curr_run[1],
                    test_version    = curr_run[2],
                    top_dir         = test_dir
                ), 'models'
            )
            if os.path.exists(pth):
                for epoch in os.listdir(pth): 
                    if args.epoch and args.epoch not in epoch:
                        continue
                    
                    curr_run += os.path.splitext(epoch)[0],
                    
                    if args.all_models:
                        runs.append(curr_run)
                    else :
                        if args.model_hash and curr_run[0] not in args.model_hash:
                            continue
                        
                        if args.train_hash and curr_run[1] not in args.train_hash:
                            continue
                        
                        if args.test_version and curr_run[2] not in args.test_version:
                            continue
                        
                        runs.append(curr_run)
    
    if not runs:
        print('No trained models found. Run train_model.py first.')
        sys.exit(0)

    # ------ Quantize each run
    for run in runs:
        training_subdir = get_training_subdir(
            training_stage  = args.stage,
            model_hash      = run[0],
            train_hash      = run[1],
            test_version    = run[2],
            top_dir         = test_dir
        )
        run_path        = os.path.join(
            training_subdir,
            'models',
            run[-1]
        )
        train_config = get_train_config_path(
            training_stage  = args.stage,
            train_hash      = run[1],
            test_version    = run[2],
            top_dir         = test_dir
        )
        model_config = get_model_config_path(
            training_stage  = args.stage,
            model_hash      = run[0],
            test_version    = run[2],
            top_dir         = test_dir
        )
        print(f'-- Quantizing: {training_subdir}')
        print(f'  -- Epoch: {args.epoch}')

        # Reconstruct and load the full floating-point model
        train_config = load_config(train_config, locals=get_locals(extract_statistics))
        model_config = load_config(model_config, locals=get_locals(custom_model))
        
        # Instantiate models
        img_enc = instantiate(model_config,'Image Encoder')
        heads   = {
            head : instantiate(model_config, head)
                for head in model_config['heads']
        }
        model   = build_model(
            img_enc = img_enc,
            **heads,
        )
        try :
            load_model(model, run_path)
            model.eval()
        except Exception as e:
            print('  -- Failed')
            if args.warn:
                warn(str(e))
            continue
        
        if args.fit_model:
            df = normalize_dataset(expand_df_labels(build_dataset())).head(512)
            calib_dataset = SkinCancerDataset(
                df,
                input_cols      = model_config['input'],
                output_cols     = model_config['output'],
                input_img_dims  = model_config['input_img_dim'],
                path_cols       = model_config['path_cols'],
                extra_transforms= get_extra_transforms(0.2, *model_config['input_img_dim'][-2:]),
                flatten         = model_config['flatten'] if 'flatten' in model_config.keys() else False,
                return_key      = False,
                return_type     ='dict',
                seed            = train_config['seed']
            )
            calib_loader    = DataLoader(calib_dataset, batch_size=512)
            calib_data, *_  = next(iter(calib_loader))
            calib_data      = to(calib_data, device)
        else :
            calib_data = None
        
        for head_name, head in heads.items():
            if args.head and head_name not in args.head:
                print('-- Skipping:', head_name)
                continue
            
            print(f'  -- Head: {head_name}')
            try :
                model_fp = fuse_faster_kan(
                    img_enc,
                    head
                )
            except Exception as e:
                print('  -- Failed')
                if args.warn:
                    warn(str(e))
                continue

            # Quantization to the KAN submodule only
            if isinstance(model_fp, torch.nn.Sequential) and 'kan' in model_fp._modules.keys():
                kan_fp = model_fp.kan
            else :
                kan_fp = model_fp

            # Build the fixed-point quantized KAN
            quant_kan = FixedPointFasterKAN(
                model           = kan_fp,
                dtype_dict      = QUANT_DTYPE,
                frac_bits_dict  = QUANT_FRAC_BITS,
                hardtanh        = args.hardtanh,
                debug           = False,
            ).to(device)

            per_layer_fitted = []
            if args.fit_model:
                print('    -- Fitting frac_bits from calibration data ...')
                quant_kan.fit_quantize(calib_data, kan_fp)

                # Collect per-layer frac_bits produced by fit, then apply the
                # per-key minimum across all layers as the global frac_bits.
                per_layer_fitted = [dict(layer.frac_bits_dict) for layer in quant_kan.layers]
                all_keys = {k for d in per_layer_fitted for k in d}
                min_frac_bits = {k: min(d[k] for d in per_layer_fitted if k in d) for k in all_keys}

                # grid, input data, and result must share the same fixed-point format
                if 'grid' in min_frac_bits and 'result' in min_frac_bits:
                    shared = min(min_frac_bits['grid'], min_frac_bits['result'])
                    min_frac_bits['grid'] = min_frac_bits['result'] = shared

                # print(f'  [DEBUG] Per-layer fitted frac_bits: {per_layer_fitted}')
                # print(f'  [DEBUG] Applying global min frac_bits: {min_frac_bits}')

                # Update quantizer with the global min frac_bits before quantization
                QUANT_FRAC_BITS.update(min_frac_bits)
                for layer in quant_kan.layers:
                    layer.update_frac_bits(min_frac_bits)
                quant_kan.quantize(kan_fp)
            else:
                quant_kan.quantize(kan_fp)

            # Save quantized model weights
            models_dir = os.path.join(
                training_subdir, 
                'quantized_models', 
                ('mixed' if args.mixed else 'int') + f'{args.num_bits}',
                head_name,
                args.epoch,
            )
            os.makedirs(models_dir, exist_ok=True)
            out_path = os.path.join(models_dir, f'{args.epoch}.pt')
            
            print(f'      >> Saved quantized model to: "{save_model(quant_kan, out_path)}"')

            # Export quantized parameters to binary files for hardware deployment
            # print(quant_kan.state_dict())
            packetize_model_to_bin(
                quant_kan.state_dict(),
                models_dir,
                DATA_CHANNELS = args.data_channels,
                RSLT_CHANNELS = args.rslt_channels,
            )
            print(f'      >> Saved bin files to:')
            print(f'        "{models_dir}/extracted_params/"')
            print(f'        "{models_dir}/packetized_params/"')

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
            print(f'      >> Saved quant config to:')
            print(f'        "{save_dict(quant_cfg, os.path.join(models_dir, 'quant_config'))}"')

            # Quick validation: compare float and quantized outputs on calib data
            if calib_data is not None:
                print('    -- Validating quantized model outputs on calibration data...')
                with torch.no_grad():
                    out_fp   = kan_fp(calib_data)
                    out_q    = FloatWrapperModule(quant_kan)(calib_data)
                    max_err  = ((out_fp - out_q)**2).max().sqrt().item()
                    mean_err = ((out_fp - out_q)**2).mean().sqrt().item()
                print(f'        -- Max output error:  {max_err:.6f}')
                print(f'        -- Mean output error: {mean_err:.6f}')
            # else:
            #     print('  (Skipping error validation, no calibration data with --no-fit)')

        # print("\n")
