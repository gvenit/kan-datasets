#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the CIFAR-100 Dataset.'
    )

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
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--no-normalize-rbf', dest='normalize_rbf', action='store_false', default=True)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--dropout-linear', dest='dropout_linear', type=float, default=None)
    parser.add_argument('--patience', dest='patience', default=10)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--lr-factor', dest='lr_factor', type=float, default=0.5)
    parser.add_argument('--lr-patience', dest='lr_patience', type=int, default=8)
    parser.add_argument('--resize', dest='resize', type=int, nargs=2, metavar=('H', 'W'), help="Resize images to HxW (e.g., --resize 16 16)")
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch
    import torchmetrics
    import hashlib
    from collections import OrderedDict

    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.callbacks import *
    from kan_utils.models import LambdaModule, FasterKAN, RSWAFF
    from kan_utils.utils import uses_momentum
    from prepare_dataset import get_class_names

    # CIFAR-100: Calculate input features based on image size
    if args.resize is not None:
        h, w = args.resize
        num_features = h * w * 3  # 3 channels (RGB)
    else:
        num_features = 3072  # 32x32x3 default
    features = [f'pixel_{i}' for i in range(num_features)]
    
    # CIFAR-100: 100 class labels
    labels = get_class_names()
    model_config = {}
    model_config['input']  = features
    model_config['output'] = labels
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
                            *([] if args.hidden_layers is None else args.hidden_layers),
                            (len(model_config['output']) if len(model_config['output']) != 2 else 1),
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
                            0,
                        ),
                        dropout_linear    = None if args.dropout_linear is None else object_to_config(
                            UpdatableFloat,
                            args.dropout_linear,
                        ),
                    ),
                ],[
                    'actf',
                    torch.nn.Sigmoid,
                    # RSWAFF,
                ]
            ]),
            target_name       = 'model',
    ))
    model_config['outputs_logits'] = False
    model_config['flatten'] = True
    train_config = get_default_training_config()
    train_config['task'] = 'mulilabel'
    train_config['sampler'] = ['Label']
    train_config['splits'] = [0.9,0.1]  
    train_config.update(
        object_to_config(
            CombinedLoss,
            object_to_config(
                WeightedLoss,
                torch.nn.BCELoss,
            ),
            object_to_config(
                WeightedLoss,
                torch.nn.MSELoss,
            ),
            target_name     = 'criterion',
    ))
    train_config['epochs'] = args.epochs
    train_config['patience'] = args.patience
    train_config['sampler'] = ['Label']
    train_config.update(
        object_to_config(
            getattr(torch.optim, args.optimizer),
            target_name     = 'optimizer',
            weight_decay    = args.weight_decay,
            amsgrad         = True,
            **({
                'momentum' : args.momentum
            } if uses_momentum(args.optimizer) else {})
    ))
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
    train_config['probability'] = 0.25
    if args.resize is not None:
        train_config['resize'] = tuple(args.resize)
    train_config['eval_criteria'] = {
        **object_to_config(
            torchmetrics.Accuracy,
            task            = 'binary',
            target_name     = 'Accuracy',
        ),
        **object_to_config(
            torchmetrics.F1Score,
            task            = 'binary',
            target_name     = 'F1Score',
        ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.PrecisionRecallCurve,
                task            = 'binary',
                thresholds      = 100,
                normalization   = False,
            ),
            **object_to_config(
                LambdaModule,
                'lambda target: target.to(torch.int8)',
                target_name     = 'targ_apply',
            ),
            target_name     = 'PrecisionRecallCurve',
        ),
        **object_to_config(
            torchmetrics.AUROC,
            task            = 'binary',
            thresholds      = 100,
            target_name     = 'AUROC',
        ),
    }
    train_config['callbacks_arguments'].update({
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            export_path = os.path.join(THIS_DIR,'dataset','tr_statistics.csv'),
            target_name = 'train_gatherer',
        ),
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            export_path = os.path.join(THIS_DIR,'dataset','val_statistics.csv'),
            overwrite   = 1,
            target_name = 'val_gatherer',
        ),
    })
    train_config['callbacks']['epoch_start'].extend([
        object_to_config(
            'lambda *args, dataloader = None, **kwargs: hasattr(dataloader.dataset,"shuffle") and '
            'dataloader.dataset.shuffle()'
        ),
    ])
    def build_test_dir(train_config, model_config, top_dir = None, test_version = None):
        pdir = os.path.join(
            '_'.join(['_'.join([key, str(val)]) for key, val in train_config.items()]),
            '_'.join(['_'.join([key, str(val)]) for key, val in model_config.items()]),
        )
        hashed = hashlib.sha1(pdir.encode()).hexdigest()
        pdir = hashed
        if top_dir is not None:
            pdir = os.path.join(top_dir,pdir)
        if test_version is not None:
            pdir = os.path.join(pdir,'_'.join(['test',test_version]))
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
