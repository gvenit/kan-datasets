#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Simplified logits configuration for HAM10000 Skin Cancer Dataset.')
    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--use-v2', dest='use_v2', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1)
    parser.add_argument('--dropout-linear', dest='dropout_linear', type=float, default=None)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--no-normalize-rbf', dest='normalize_rbf', action='store_false', default=True)
    parser.add_argument('--patience', dest='patience', default=50)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--lr', dest='lr', type=float, default=3e-4)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='AdamW')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--lr-factor', dest='lr_factor', type=float, default=0.75)
    parser.add_argument('--lr-patience', dest='lr_patience', type=int, default=8)
    parser.add_argument('--resize', dest='resize', type=int, nargs=2, metavar=('H', 'W'), help="Resize images to HxW (e.g., --resize 64 64)")
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save configs")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch
    import torchmetrics
    import hashlib
    from collections import OrderedDict
    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.callbacks import *
    from kan_utils.models import *
    from kan_utils.utils import uses_momentum, expand_value
    from prepare_dataset import build_dataset, expand_df_labels

    # Get dataset and labels
    df = expand_df_labels(build_dataset())
    labels = [col for col in df.columns if 'Lesion_Is_' in col]

    # Calculate input features based on image size
    if args.resize is not None:
        h, w = args.resize
        num_features = h * w * 3  # RGB
    else:
        num_features = 64*64*3
    features = [f'pixel_{i}' for i in range(num_features)]

    # Model configuration
    model_config = {}
    model_config['input']  = features
    model_config['output'] = labels
    model_config['input_img_dim'] = (3,) + (tuple(args.resize) if args.resize else (64, 64))
    model_config['path_cols'] = {'image': ['Image']}
    model_config['flatten'] = True

    args.hidden_layers = [] if args.hidden_layers is None else args.hidden_layers
    args.num_grids = expand_value(args.num_grids, len(args.hidden_layers) + 1) if args.num_grids else None
    args.grid_min = expand_value(args.grid_min, len(args.hidden_layers) + 1) if args.grid_min else None
    args.grid_max = expand_value(args.grid_max, len(args.hidden_layers) + 1) if args.grid_max else None
    args.scale = expand_value(args.scale, len(args.hidden_layers) + 1) if args.scale else None

    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            object_to_config(
                OrderedDict, [[
                    'kan',
                    object_to_config(
                        FasterKAN,
                        hidden_layers     = [
                            len(model_config['input']),
                            *(args.hidden_layers),
                            len(labels),
                        ],
                        num_grids         = args.num_grids,
                        grid_min          = args.grid_min,
                        grid_max          = args.grid_max,
                        inv_denominator   = args.scale,
                        mode              = args.mode,
                        residual          = args.residual,
                        dynamic           = args.dynamic,
                        use_v2            = args.use_v2,
                        normalize         = args.normalize,
                        normalize_rbf     = args.normalize_rbf,
                        dropout_rate      = object_to_config(
                            UpdatableFloat,
                            args.dropout,
                        ),
                        dropout_linear    = None if args.dropout_linear is None else object_to_config(
                            UpdatableFloat,
                            args.dropout_linear,
                        ),
                    ),
                ],[
                    'dropout', object_to_config(
                        torch.nn.Dropout,
                        object_to_config(
                            UpdatableFloat,
                            args.dropout,
                        )
                    ),
                ]
            ]),
            target_name       = 'model',
    ))
    model_config['outputs_logits'] = True

    # Training configuration
    train_config = get_default_training_config()
    train_config['task'] = 'multiclass'
    train_config['sampler'] = labels
    train_config['splits'] = [0.9, 0.1]

    # Loss: BCEWithLogitsLoss 
    train_config.update(
        object_to_config(
            torch.nn.BCEWithLogitsLoss,
            target_name = 'criterion',
    ))

    train_config['epochs'] = args.epochs
    train_config['patience'] = args.patience

    # Optimizer
    if args.optimizer == 'RMSProp':
        train_config.update(
            object_to_config(
                torch.optim.RMSprop(),
                target_name     = 'optimizer',
                alpha           = 0.75,
                centered        = True,
                weight_decay    = args.weight_decay,
                momentum        = args.momentum,
        ))
    else:
        train_config.update(
            object_to_config(
                getattr(torch.optim, args.optimizer),
                target_name     = 'optimizer',
                weight_decay    = args.weight_decay,
                **({
                    'momentum' : args.momentum
                } if uses_momentum(args.optimizer) else {})
        ))

    # Scheduler
    train_config.update(
        object_to_config(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            factor      = args.lr_factor,
            patience    = args.lr_patience,
            target_name = 'scheduler'
    ))

    train_config['lr'] = args.lr
    train_config['seed'] = args.seed
    train_config['batch_size'] = args.batch_size

    # Evaluation metrics
    train_config['eval_criteria'] = {
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.Accuracy,
                task            = 'multiclass',
                num_classes     = len(labels),
            ),
            targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
            target_name     = 'Accuracy',
        ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.F1Score,
                task            = 'multiclass',
                num_classes     = len(labels),
            ),
            targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
            target_name     = 'F1Score',
        ),
        # **object_to_config(
        #     ProcessAndApplyMetric,
        #     object_to_config(
        #         torchmetrics.AUROC,
        #         task            = 'multiclass',
        #         num_classes     = len(labels),
        #         thresholds      = 100,
        #     ),
        #     targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
        #     target_name     = 'AUROC',
        # ),
    }

    # Build directory/hash
    def build_test_dir(train_config, model_config, top_dir = None, test_version = None):
        pdir = os.path.join(
            '_'.join(['_'.join([key, str(val)]) for key, val in train_config.items()]),
            '_'.join(['_'.join([key, str(val)]) for key, val in model_config.items()]),
        )
        hashed = hashlib.sha1(pdir.encode()).hexdigest()
        pdir = hashed
        if top_dir is not None:
            pdir = os.path.join(top_dir, pdir)
        if test_version is not None:
            pdir = os.path.join(pdir, '_'.join(['test', test_version]))
        return pdir, hashed

    pdir, hashed = build_test_dir(train_config, model_config, top_dir=args.dest_top_dir, test_version=args.test_version)

    if not args.export :
        print(f'Test directory : {pdir}')

    if args.hash:
        print(hashed)

    if args.export :
        path = os.path.join(pdir,'config','train')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = save_config(train_config, path)

        path = os.path.join(pdir,'config','model')
        path = save_config(model_config, path)

    if not args.hash:
        print(os.path.dirname(pdir))
