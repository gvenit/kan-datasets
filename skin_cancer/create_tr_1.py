#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__' :
    parser = ArgumentParser(
        description='Training configuration script for 1st training stage training the Alzheimer\'s Dataset.'
    )

    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--patience', dest='patience', default=0)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the training configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch
    import torchmetrics

    from kan_utils.config import *
    from kan_utils.metrics import *
    # from kan_utils.callbacks import FlattenBatch
    from kan_utils.utils import uses_momentum
    from kan_utils.models import LambdaModule
    
    from prepare_dataset import build_dataset, expand_df_labels
    from build_model import build_training_dir

    args.test_version = '_'.join(['test',args.test_version])

    df = expand_df_labels(build_dataset())

    train_config = get_default_training_config()
    train_config.update(
        object_to_config(
            MultiHeadLoss,
            expect_type = 'dict',
            reduction   = 'sum',
            Segmentor = object_to_config(
                CombinedLoss,
                torch.nn.MSELoss,
                torch.nn.L1Loss,
                torch.nn.BCELoss,
                # object_to_config(
                #     Accuracy2Loss,
                #     target          = type_to_config(torchmetrics.image.StructuralSimilarityIndexMeasure),
                #     instantiated    = False,
                #     data_range      = 1.0,
                #     # **object_to_config(                               # In order to handle images smaller than 160x160
                #     #     tuple,
                #     #     [0.0448, 0.2856, 0.3001],
                #     #     target_name = 'betas'
                #     # ),
                #     reduction       = 'elementwise_mean',
                # ),
            ),
            Classifier = object_to_config(
                CombinedLoss,
                torch.nn.MSELoss,
                torch.nn.L1Loss,
                torch.nn.BCELoss,
            ),
            target_name     = 'criterion',
        ))
    train_config['patience']    = args.patience
    train_config['epochs']      = args.epochs
    # train_config['clip_limit']  = 1.
    train_config['sampler']     = df.drop(columns=['Image','Mask']).columns.tolist()
    train_config.update(
        object_to_config(
            getattr(torch.optim, args.optimizer),
            target_name     = 'optimizer',
            weight_decay    = args.weight_decay,
            **({
                'momentum' : args.momentum
            } if uses_momentum(args.optimizer) else {})
    ))
    train_config.update(
        object_to_config(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            # factor      = 0.1,
            factor      = 0.5,
            # patience    = 10,
            patience    = 8,
            target_name = 'scheduler'
    ))
    train_config['lr'] = args.lr
    train_config['seed'] = args.seed
    train_config['batch_size'] = args.batch_size
    train_config['exclude_groups'] = [
        ["Lesion (6)",],
    ]
    train_config['eval_criteria'] = {
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                LambdaModule,
                'lambda x, y : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std())',
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                LambdaModule,
                'lambda x, y : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std())',
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Stats -- Predictions',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                LambdaModule,
                'lambda y, x : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std())',
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                LambdaModule,
                'lambda y, x : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std())',
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Stats -- Targets',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torch.nn.BCELoss,
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                torch.nn.BCELoss,
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'BCELoss',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torch.nn.MSELoss,
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                torch.nn.MSELoss,
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'MSE',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torch.nn.L1Loss,
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                torch.nn.L1Loss,
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'MAE',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Accuracy,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'sum',
            expect_type     = 'dict',
            target_name     = 'Accuracy -- Global',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Accuracy,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='none',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Accuracy -- Label-wise',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Recall,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Recall -- Global',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Recall,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='none',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Recall -- Label-wise',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.F1Score,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'F1Score -- Global',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.F1Score,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='none',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'F1Score -- Label-wise',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Precision,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Precision -- Global',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Precision,
                task                = 'multilabel',
                multidim_average    ='global',
                average             ='none',
                num_labels          = len(df.drop(columns=['Image','Mask']).columns),
                target_name         = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Precision -- Label-wise',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.image.PeakSignalNoiseRatio,
                target_name     = 'Segmentor',
                data_range      = 1.0,
                reduction       = 'elementwise_mean',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'PSNR',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.image.StructuralSimilarityIndexMeasure,
                target_name     = 'Segmentor',
                data_range      = 1.0,
                # **object_to_config(                               # In order to handle images smaller than 160x160
                #     tuple,
                #     [0.0448, 0.2856, 0.3001],
                #     target_name = 'betas'
                # ),
                reduction       = 'elementwise_mean',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'SSIM',
        ),
    }
    # train_config['callbacks']['train_iter_start'].append(
    #     object_to_config(
    #         FlattenBatch,
    #         data_dim    = -3
    #     )
    # )
    # train_config['callbacks']['eval_iter_start'].append(
    #     object_to_config(
    #         FlattenBatch,
    #         data_dim    = -3
    #     )
    # ) 
    pdir, train_config['hash'] = build_training_dir(
        train_config, 
        top_dir         = args.dest_top_dir, 
        test_version    = args.test_version,
        training_stage  = 1,
    )
    if args.hash:
        print(train_config['hash'])

    if args.export :
        os.makedirs(os.path.dirname(pdir), exist_ok=True)
        pdir = save_config(train_config, pdir)
        
    if not args.hash:
        print(pdir)
