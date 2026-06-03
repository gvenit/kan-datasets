#!/usr/bin/env python3
"""
Quantized FasterKAN Validation Script - MNIST

Two-phase script:
  Phase 1 - Quantize: loads each trained FasterKAN, applies fixed-point
             quantization, saves the quantized KAN weights and binary exports.
  Phase 2 - Validate: loads each saved quantized model, wraps it with
             FloatWrapperModule, and evaluates it on the test split.

Usage (run from mnist/ or any folder that has prepare_dataset.py /
custom_dataset.py on the Python path):

    python validate_custom_quant.py [options]

    -d / --train-dir   Top-level training output dir  (default: ./train)
    --bits             Base bit-width                  (default: 16)
    --mixed            Use mixed bit-width (weight = bits/2)
    --signed-actf      Use signed activation dtype
    --no-fit           Skip calibration-based frac_bits fitting
    --hardtanh         Integer Hardtanh approx (RSWAFF mode only)
    --epoch            Checkpoint name                 (default: best)
    --skip-quant       Skip Phase 1, only validate pre-saved quantized models
    --no-pbar          Suppress progress bars
    --data-channels    Parallel input channels         (default: 1)
    --rslt-channels    Outputs per iteration           (default: 1)
    -m / --model       Process a single run by folder name
"""

if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.insert(0, TOP_DIR)
    sys.path.insert(0, THIS_DIR)

    parser = ArgumentParser(description='Quantize and validate FasterKAN models for MNIST.')
    parser.add_argument('-d', '--train-dir',    dest='train_dir',     default=os.path.join(THIS_DIR, 'train'))
    parser.add_argument('--bits',               dest='num_bits',      type=int,  default=16)
    parser.add_argument('--data-channels',      dest='data_channels', type=int,  default=1)
    parser.add_argument('--rslt-channels',      dest='rslt_channels', type=int,  default=1)
    parser.add_argument('--no-fit',             dest='fit_model',     action='store_false', default=True)
    parser.add_argument('--hardtanh',           dest='hardtanh',      action='store_true',  default=False)
    parser.add_argument('--mixed',              dest='mixed',         action='store_true',  default=False)
    parser.add_argument('--signed-actf',        dest='signed_actf',   action='store_true',  default=False)
    parser.add_argument('--skip-quant',         dest='skip_quant',    action='store_true',  default=False)
    parser.add_argument('--no-pbar',            dest='no_pbar',       action='store_true',  default=False)
    parser.add_argument('--epoch',              dest='epoch',         type=str,  default='best')
    parser.add_argument('--verbose',            dest='verbose',       type=int,  default=1)

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all',  dest='all_models',  action='store_true', default=False)
    sel.add_argument('-m', '--model', dest='model_name', type=str, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from kan_utils.config.config  import load_config, instantiate, weak_instantiate_all
    from kan_utils.utils          import load_model, save_model, save_dict, load_dict, set_seed
    from kan_utils.quantization   import FixedPointFasterKAN, FloatWrapperModule
    from kan_utils.quantization.parameter_transform import packetize_model_to_bin
    from kan_utils.training       import evaluate

    from prepare_dataset  import build_dataset, get_dataset, normalize_data
    from custom_dataset   import MNISTDataset

    # -------- Quantization configuration
    NUM_BITS_LOW  = args.num_bits // (1 + int(args.mixed))
    NUM_BITS_HIGH = args.num_bits

    QUANT_DTYPE = {
        'grid'   : (NUM_BITS_HIGH,  False),
        'scale'  : (NUM_BITS_HIGH,  True),
        'weight' : (NUM_BITS_LOW,   False),
        'sdff'   : (NUM_BITS_HIGH,  False),
        'actf'   : (NUM_BITS_HIGH,  not args.signed_actf),
        'result' : (NUM_BITS_HIGH,  False),
    }
    QUANT_FRAC_BITS = {
        'grid'   : NUM_BITS_HIGH // 2,
        'scale'  : NUM_BITS_HIGH // 2,
        'weight' : NUM_BITS_LOW  // 2,
        'sdff'   : NUM_BITS_HIGH // 2,
        'actf'   : NUM_BITS_HIGH,
        'result' : NUM_BITS_HIGH // 2,
    }

    device = torch.device('cpu')

    # ------ Collect all training runs
    train_dir = args.train_dir
    runs = []
    for hash_name in sorted(os.listdir(train_dir)):
        hash_path = os.path.join(train_dir, hash_name)
        if not os.path.isdir(hash_path):
            continue
        for run_name in sorted(os.listdir(hash_path)):
            run_path         = os.path.join(hash_path, run_name)
            best_model_path  = os.path.join(run_path, 'models', f'{args.epoch}.pt')
            model_cfg_path   = os.path.join(run_path, 'config', 'model.json')
            train_cfg_path   = os.path.join(run_path, 'config', 'train.json')
            quant_dir        = os.path.join(run_path, 'quantized_models', args.epoch)
            quant_model_path = os.path.join(quant_dir, f'{args.epoch}_quantized.pt')

            if not os.path.exists(best_model_path):
                continue
            runs.append({
                'run_path'         : run_path,
                'best_model_path'  : best_model_path,
                'quant_model_path' : quant_model_path,
                'quant_dir'        : quant_dir,
                'model_cfg_path'   : model_cfg_path,
                'train_cfg_path'   : train_cfg_path,
            })

    if not runs:
        if args.verbose > 0:
            print('No trained models found. Run train_model.py first.')
        sys.exit(0)

    # ------ Filter by --model
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

    # ------ Build shared DataLoaders
    first_model_cfg = load_config(runs[0]['model_cfg_path'])
    first_train_cfg = load_config(runs[0]['train_cfg_path'])
    set_seed(first_train_cfg['seed'])

    build_dataset()

    # Build preprocessing function (with optional resize from train config)
    resize = first_train_cfg.get('resize', None)
    if resize is not None:
        resize = tuple(int(r) for r in resize)
        def _preprocess(images):
            t = torch.from_numpy(np.array(images, dtype=np.float32)).unsqueeze(1)
            t = F.interpolate(t, size=resize, mode='bilinear', align_corners=False)
            return normalize_data(t.squeeze(1).numpy())
    else:
        def _preprocess(images):
            return normalize_data(images)

    calib_data = None
    if args.fit_model and not args.skip_quant:
        calib_raw, calib_labels = get_dataset('train_val')
        calib_dataset = MNISTDataset(
            data            = calib_raw,
            labels          = calib_labels,
            task            = first_train_cfg['task'],
            preprocess_data = _preprocess,
            flatten         = first_model_cfg.get('flatten', True),
        )
        calib_loader = DataLoader(calib_dataset, batch_size=512)
        calib_data, _ = next(iter(calib_loader))
        calib_data = calib_data.to(device)

    test_raw, test_labels = get_dataset('test')
    test_dataset = MNISTDataset(
        data            = test_raw,
        labels          = test_labels,
        task            = first_train_cfg['task'],
        return_key      = True,
        preprocess_data = _preprocess,
        flatten         = first_model_cfg.get('flatten', True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = first_train_cfg['batch_size'],
        num_workers = os.cpu_count(),
        pin_memory  = False,
    )

    # ================================================================
    # PHASE 1 - Quantize
    # ================================================================
    if not args.skip_quant:
        print('\n========== PHASE 1: Quantization ==========')
        for run in runs:
            run_path         = run['run_path']
            best_model_path  = run['best_model_path']
            model_cfg_path   = run['model_cfg_path']
            quant_dir        = run['quant_dir']
            quant_model_path = run['quant_model_path']

            print(f'\n---- Quantizing: {run_path},  Epoch: {args.epoch} ----')

            model_config = load_config(model_cfg_path)
            model_fp     = instantiate(model_config, 'model').to(device)
            state_dict   = torch.load(best_model_path, map_location=device)
            model_fp.load_state_dict(state_dict)
            model_fp.eval()

            kan_fp    = model_fp.kan
            quant_kan = FixedPointFasterKAN(
                model          = kan_fp,
                dtype_dict     = QUANT_DTYPE,
                frac_bits_dict = QUANT_FRAC_BITS,
                hardtanh       = args.hardtanh,
            ).to(device)

            if args.fit_model:
                print('  Fitting frac_bits from calibration data ...')
                quant_kan.fit_quantize(calib_data, kan_fp)

                per_layer_fitted = [dict(layer.frac_bits_dict) for layer in quant_kan.layers]
                all_keys = {k for d in per_layer_fitted for k in d}
                min_frac_bits = {k: min(d[k] for d in per_layer_fitted if k in d) for k in all_keys}

                if 'grid' in min_frac_bits and 'result' in min_frac_bits:
                    shared = min(min_frac_bits['grid'], min_frac_bits['result'])
                    min_frac_bits['grid'] = min_frac_bits['result'] = shared

                for layer in quant_kan.layers:
                    layer.update_frac_bits(min_frac_bits)
                quant_kan.quantize(kan_fp)
            else:
                per_layer_fitted = []
                quant_kan.quantize(kan_fp)

            os.makedirs(quant_dir, exist_ok=True)
            save_model(quant_kan, quant_model_path, device)
            print(f'  Saved quantized model → {quant_model_path}')

            packetize_model_to_bin(
                quant_kan.state_dict(),
                quant_dir,
                DATA_CHANNELS = args.data_channels,
                RSLT_CHANNELS = args.rslt_channels,
            )
            print(f'  Saved bin files → {quant_dir}/extracted_params/  {quant_dir}/packetized_params/')

            quant_cfg = {
                'num_bits_low'        : NUM_BITS_LOW,
                'num_bits_high'       : NUM_BITS_HIGH,
                'hardtanh'            : args.hardtanh,
                'fit_model'           : args.fit_model,
                'dtype_dict'          : {k: list(v) for k, v in QUANT_DTYPE.items()},
                'frac_bits_dict'      : QUANT_FRAC_BITS,
                'per_layer_frac_bits' : [layer.frac_bits_dict for layer in quant_kan.layers],
            }
            save_dict(quant_cfg, os.path.join(quant_dir, 'quant_config'))
            print(f'  Saved quant config → {quant_dir}/quant_config.json')

            if calib_data is not None:
                with torch.no_grad():
                    out_fp   = kan_fp(calib_data)
                    out_q    = FloatWrapperModule(quant_kan)(calib_data)
                    max_err  = (out_fp - out_q).abs().max().item()
                    mean_err = (out_fp - out_q).abs().mean().item()
                print(f'  Calibration error - max: {max_err:.6f}  mean: {mean_err:.6f}')

    # ================================================================
    # PHASE 2 - Validate
    # ================================================================
    print('\n========== PHASE 2: Validation ==========')
    for run in runs:
        run_path         = run['run_path']
        quant_model_path = run['quant_model_path']
        model_cfg_path   = run['model_cfg_path']
        train_cfg_path   = run['train_cfg_path']
        quant_dir        = run['quant_dir']

        if not os.path.exists(quant_model_path):
            print(f'[SKIP] No quantized model at {quant_model_path}')
            continue

        print(f'\n---- Validating: {run_path},  Epoch: {args.epoch} ----')

        model_config = load_config(model_cfg_path)
        train_config = load_config(train_cfg_path)

        eval_criteria = {**weak_instantiate_all(train_config['eval_criteria'])}
        if 'loss' not in eval_criteria:
            eval_criteria['loss'] = instantiate(train_config, 'criterion')

        # Load the trained float model
        model_fp    = instantiate(model_config, 'model').to(device)
        float_state = torch.load(run['best_model_path'], map_location=device)
        model_fp.load_state_dict(float_state)
        model_fp.eval()
        kan_fp = model_fp.kan

        # Quick float-model sanity check on one test batch
        with torch.no_grad():
            sample_x, sample_y, *_ = next(iter(test_loader))
            sample_x = sample_x.to(device)
            out_fp = kan_fp(sample_x)
            print(f'  [diag] float KAN output range: [{out_fp.min():.4f}, {out_fp.max():.4f}]  std={out_fp.std():.4f}')

        # Reconstruct quantized shell
        quant_kan = FixedPointFasterKAN(model=kan_fp).to(device)
        load_model(quant_kan, quant_model_path, device)
        quant_kan.eval()

        # Sanity check quantized output
        with torch.no_grad():
            out_q = FloatWrapperModule(quant_kan)(sample_x)
            print(f'  [diag] quant  KAN output range: [{out_q.min():.4f}, {out_q.max():.4f}]  std={out_q.std():.4f}')
            print(f'  [diag] max|fp - q| = {(out_fp - out_q).abs().max():.6f}')
            print('  [diag] === layer-by-layer debug (8 samples) ===')
            _ = quant_kan(sample_x[:8], _debug=False)

        wrapped = FloatWrapperModule(quant_kan)

        rslt_dir = os.path.join(quant_dir, 'rslt')
        os.makedirs(rslt_dir, exist_ok=True)
        checkpoint_path = os.path.join(rslt_dir, args.epoch)

        metrics = evaluate(
            wrapped,
            eval_dataloader = test_loader,
            criteria        = eval_criteria,
            keep_copy       = True,
            checkpoint_path = checkpoint_path,
            epoch           = args.epoch,
            sample_weight   = train_config.get('sample_weight', False),
            show_pbar       = not args.no_pbar,
            device          = device,
        )

        print('\n -- Evaluation results --')
        print(f'  Accuracy: {float(metrics.get("Accuracy", "N/A")):.4f}  Loss: {float(metrics.get("loss", "N/A")):.4f}')
        print(f'  F1Score: {float(metrics.get("F1Score", "N/A")):.4f}')
        print(f'  AUROC: {float(metrics.get("AUROC", "N/A")):.4f}\n')

        # Persist into history.json (quantized key)
        hist_path = os.path.join(run_path, 'history')
        try:
            history = load_dict(hist_path)
        except (FileNotFoundError, KeyError):
            history = {}
        history.setdefault('quantized', {})[args.epoch] = metrics
        save_dict(history, hist_path)
        print(f'  History saved → {hist_path}.json')
