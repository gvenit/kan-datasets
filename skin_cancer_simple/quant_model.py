#!/usr/bin/env python3
"""
Fixed-Point Quantization Script

Loads each trained FasterKAN model from train/, applies
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
    import sys
    import os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.insert(0, TOP_DIR)

    parser = ArgumentParser(description='Quantize trained FasterKAN models for the Skin Cancer dataset.')
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR, 'train'),
                        help='The directory to be used as a top directory for training.')
    parser.add_argument('--bits', dest='num_bits', type=int, default=16,
                        help='Base bit-width for quantization (default: 16)')
    parser.add_argument('--data-channels', dest='data_channels', type=int, default=1,
                        help='Parallel input channels (default: 1)')
    parser.add_argument('--rslt-channels', dest='rslt_channels', type=int, default=1,
                        help='Outputs per iteration (default: 1)')
    parser.add_argument('--no-fit', dest='fit_model', action='store_false', default=True,
                        help='Skip calibration-based frac_bits fitting')
    parser.add_argument('--hardtanh', dest='hardtanh', action='store_true', default=False,
                        help='Use integer Hardtanh approximation (RSWAFF mode only)')
    parser.add_argument('--mixed', dest='mixed', action='store_true', default=False,
                        help='Use the mixedDW representation')
    parser.add_argument('--signed-actf', dest='signed_actf', action='store_true', default=False,
                        help='Use signed activation function')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best',
                        help='Which checkpoint to quantize (e.g., "best", "epoch_100", default: "best")')
    parser.add_argument('--stage', dest='stage', type=int, default=1,
                        help='Training stage (ignored for the given structure)')
    parser.add_argument('-w', '--warn', dest='warn', action='store_true',
                        help='Show warning messages')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False,
                     help='Process all models in the train directory (default behaviour)')

    group = sel.add_argument_group()
    group.add_argument('-m', '--model', dest='model_hash', type=str, nargs='*',
                       help='Process a specific model by hash')
    group.add_argument('-t', '--train', dest='train_hash', type=str, nargs='*',
                       help='Process a specific training by hash')
    group.add_argument('-v', '--version', '--test-version', dest='test_version', type=str, nargs='*',
                       help='Process a specific training by hash')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all-heads', dest='all_heads', action='store_true', default=False,
                     help='Process all heads of a model (default behaviour)')
    sel.add_argument('--head', dest='head', type=str, nargs='*',
                     help='Process a specific head of a model by folder name, e.g. test_0 or <hash>')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    from warnings import warn

    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config import load_config, instantiate
    from kan_utils.models import fuse_faster_kan
    from kan_utils.utils import load_model, save_model, save_dict, to
    from kan_utils.quantization import FixedPointFasterKAN, FloatWrapperModule
    from kan_utils.quantization.parameter_transform import packetize_model_to_bin

    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
    from custom_dataset import SkinCancerDataset, get_extra_transforms

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
        'sdff'   : (NUM_BITS_HIGH) // 2,
        'actf'   : NUM_BITS_HIGH,
        'result' : NUM_BITS_HIGH // 2,
    }

    device = torch.device('cpu')

    # ------ Collect all training runs from train dir
    print("------ Quantization Script for Skin Cancer Dataset ------")
    train_dir = args.test_dir
    if not os.path.isabs(train_dir):
        train_dir = os.path.join(THIS_DIR, train_dir)

    if not os.path.isdir(train_dir):
        print(f'Training directory not found: {train_dir}')
        sys.exit(1)

    # Discover all runs
    runs = []  # each run is a tuple: (model_hash, test_version, epoch, base_dir, checkpoint_path, model_config_path, train_config_path)
    for model_hash in os.listdir(train_dir):
        model_path = os.path.join(train_dir, model_hash)
        if not os.path.isdir(model_path):
            continue

        if args.model_hash and model_hash not in args.model_hash:
            continue

        for test_version in os.listdir(model_path):
            version_path = os.path.join(model_path, test_version)
            if not os.path.isdir(version_path):
                continue

            if args.test_version and test_version not in args.test_version:
                continue

            config_dir = os.path.join(version_path, 'config')
            model_config_path = os.path.join(config_dir, 'model.json')
            train_config_path = os.path.join(config_dir, 'train.json')
            if not (os.path.isfile(model_config_path) and os.path.isfile(train_config_path)):
                continue

            models_dir = os.path.join(version_path, 'models')
            if not os.path.isdir(models_dir):
                continue

            for epoch_file in os.listdir(models_dir):
                if args.epoch != 'best' and args.epoch not in epoch_file:
                    continue
                epoch_name = os.path.splitext(epoch_file)[0]
                checkpoint_path = os.path.join(models_dir, epoch_file)

                runs.append((
                    model_hash, test_version, epoch_name, version_path,
                    checkpoint_path, model_config_path, train_config_path
                ))

    if not runs:
        print('No trained models found. Run train_model.py first.')
        sys.exit(0)

    # ------ Quantize each run
    for run in runs:
        model_hash, test_version, epoch_name, base_dir, checkpoint_path, model_config_path, train_config_path = run
        print(f'-- Quantizing: {base_dir}')
        print(f'  -- Epoch: {epoch_name}')

        # Load configurations
        train_config = load_config(train_config_path)
        model_config = load_config(model_config_path)

        # ------------------------------------------------------------------
        # Build the model – two possible structures:
        #   1) Multi-head: model_config contains 'heads' key.
        #   2) Single model: model_config contains a 'model' key (the Sequential).
        # ------------------------------------------------------------------
        if 'heads' in model_config:
            # Original multi‑head architecture
            img_enc = instantiate(model_config, 'Image Encoder')
            heads = {
                head: instantiate(model_config, head)
                for head in model_config['heads']
            }
            # We will loop over heads later
            head_items = list(heads.items())
        else:
            # New single‑model architecture: instantiate directly
            model = instantiate(model_config, key='model')
            # We treat the whole model as one head named "model"
            head_items = [('model', model)]

        # Load checkpoint into the model(s)
        try:
            # For multi‑head, the checkpoint contains the entire model state dict.
            # For single‑model, we can load directly.
            if 'heads' in model_config:
                # Build the full model using build_model (which should exist in kan_utils.models)
                # If build_model is not available, we need to reconstruct the combined model.
                # Assuming build_model exists and works:
                from kan_utils.models import build_model
                full_model = build_model(img_enc=img_enc, **heads)
                load_model(full_model, checkpoint_path)
                full_model.eval()
            else:
                load_model(model, checkpoint_path)
                model.eval()
        except Exception as e:
            print('  -- Failed to load model')
            if args.warn:
                warn(str(e))
            continue

        # Prepare calibration data if fitting
        calib_data = None
        if args.fit_model:
            df = normalize_dataset(expand_df_labels(build_dataset())).head(512)
            calib_dataset = SkinCancerDataset(
                df,
                input_cols      = model_config['input'],
                output_cols     = model_config['output'],
                input_img_dims  = model_config['input_img_dim'],
                path_cols       = model_config['path_cols'],
                extra_transforms= get_extra_transforms(0.2, *model_config['input_img_dim'][-2:]),
                flatten         = model_config.get('flatten', False),
                return_key      = False,
                return_type     = 'dict',
                seed            = train_config['seed']
            )
            calib_loader = DataLoader(calib_dataset, batch_size=512)
            calib_data, *_ = next(iter(calib_loader))
            calib_data = to(calib_data, device)

        # Process each head (or the single model)
        for head_name, head in head_items:
            if args.head and head_name not in args.head:
                print('-- Skipping:', head_name)
                continue

            print(f'  -- Head: {head_name}')

            try:
                if 'heads' in model_config:
                    # Multi‑head: fuse encoder and head
                    fused = fuse_faster_kan(img_enc, head)
                else:
                    # Single‑model: already a Sequential, just get it
                    fused = head
            except Exception as e:
                print('  -- Failed to fuse model')
                if args.warn:
                    warn(str(e))
                continue

            # Extract the KAN submodule
            if isinstance(fused, torch.nn.Sequential) and hasattr(fused, 'kan'):
                kan_fp = fused.kan
            else:
                # Fallback: if no 'kan' attribute, assume the whole model is the KAN
                kan_fp = fused

            # Build fixed‑point quantized KAN
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

                # Collect per‑layer fitted frac_bits
                per_layer_fitted = [dict(layer.frac_bits_dict) for layer in quant_kan.layers]
                all_keys = {k for d in per_layer_fitted for k in d}
                min_frac_bits = {k: min(d[k] for d in per_layer_fitted if k in d) for k in all_keys}

                if 'grid' in min_frac_bits and 'result' in min_frac_bits:
                    shared = min(min_frac_bits['grid'], min_frac_bits['result'])
                    min_frac_bits['grid'] = min_frac_bits['result'] = shared

                QUANT_FRAC_BITS.update(min_frac_bits)
                for layer in quant_kan.layers:
                    layer.update_frac_bits(min_frac_bits)
                quant_kan.quantize(kan_fp)
            else:
                quant_kan.quantize(kan_fp)

            # Save quantized model and binary files
            models_dir = os.path.join(
                base_dir,
                'quantized_models',
                ('mixed' if args.mixed else 'int') + f'{args.num_bits}',
                head_name,
                epoch_name,
            )
            os.makedirs(models_dir, exist_ok=True)
            out_path = os.path.join(models_dir, f'{epoch_name}.pt')
            print(f'      >> Saved quantized model to: "{save_model(quant_kan, out_path)}"')

            packetize_model_to_bin(
                quant_kan.state_dict(),
                models_dir,
                DATA_CHANNELS=args.data_channels,
                RSLT_CHANNELS=args.rslt_channels,
            )
            print(f'      >> Saved bin files to:')
            print(f'        "{models_dir}/extracted_params/"')
            print(f'        "{models_dir}/packetized_params/"')

            # Save quantization config
            quant_cfg = {
                'num_bits_low': NUM_BITS_LOW,
                'num_bits_high': NUM_BITS_HIGH,
                'hardtanh': args.hardtanh,
                'fit_model': args.fit_model,
                'dtype_dict': {k: list(v) for k, v in QUANT_DTYPE.items()},
                'frac_bits_dict': QUANT_FRAC_BITS,
                'per_layer_frac_bits_fitted': per_layer_fitted,
                'per_layer_frac_bits_final': [layer.frac_bits_dict for layer in quant_kan.layers],
            }
            print(f'      >> Saved quant config to:')
            print(f'        "{save_dict(quant_cfg, os.path.join(models_dir, "quant_config"))}"')

            # Validation
            if calib_data is not None:
                print('    -- Validating quantized model outputs on calibration data...')
                with torch.no_grad():
                    out_fp = kan_fp(calib_data)
                    out_q = FloatWrapperModule(quant_kan)(calib_data)
                    max_err = ((out_fp - out_q) ** 2).max().sqrt().item()
                    mean_err = ((out_fp - out_q) ** 2).mean().sqrt().item()
                print(f'        -- Max output error:  {max_err:.6f}')
                print(f'        -- Mean output error: {mean_err:.6f}')