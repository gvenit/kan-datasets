#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the Ship Performance Clusterring Dataset.'
    )
    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--patience', dest='patience', default=10)
    parser.add_argument('--epochs', dest='epochs', default=500)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--lr-factor', dest='lr_factor', type=float, default=0.5)
    parser.add_argument('--lr-patience', dest='lr_patience', type=int, default=8)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch
    import torchmetrics
    import hashlib
    
    from kan_utils.config import *
    from kan_utils.metrics import *
    # from kan_utils.callbacks import *
    from kan_utils.models import LambdaModule
    from kan_utils.utils import uses_momentum
    
    from prepare_dataset import build_dataset, expand_df_labels
    from build_model import build_training_dir

    args.test_version = '_'.join(['test',args.test_version])

    df = expand_df_labels(build_dataset())
    labels = df.drop(columns=['Image','Mask']).columns.tolist()
    
    train_config = get_default_training_config()
    train_config['task'] = 'multiclass'
    train_config['splits'] = [0.9,0.1]  
    
    train_config.update(
        object_to_config(
            ProcessAndApplyMetric,
            metric      = object_to_config(
                torch.nn.CrossEntropyLoss,
                label_smoothing = 0.1,  # Disabled for testing
            ),
            pred_apply  = 'lambda pred: pred * 4',
            target_name = 'criterion',
    ))
    train_config.update(
        object_to_config(
            MultiHeadLoss,
            expect_type = 'dict',
            reduction   = 'sum',
            Segmentor = object_to_config(
                CombinedLoss,
                torch.nn.MSELoss,
                torch.nn.L1Loss,
                torch.nn.BCEWithLogitsLoss,
            ),
            Classifier = object_to_config(
                CombinedLoss,
                torch.nn.MSELoss,
                torch.nn.L1Loss,
                torch.nn.CrossEntropyLoss,
            ),
            target_name     = 'criterion',
    ))
    train_config['epochs']      = args.epochs
    train_config['patience']    = args.patience
    # train_config['clip_limit']  = 1.
    train_config['sampler']     = labels
    
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
    else :
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
            factor      = args.lr_factor,
            patience    = args.lr_patience,
            target_name = 'scheduler'
    ))
    train_config['lr'] = args.lr
    train_config['seed'] = args.seed
    train_config['batch_size'] = args.batch_size
    train_config['exclude_groups'] = [
        ["Lesion (6)",],
    ]
    # train_config['probability'] = 0.25
    
    train_config['eval_criteria'] = {
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                LambdaModule,
                'lambda x, y : { "Predictions" : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std()),'
                                '"Targets"     : dict(min = y.min(),max = y.max(),mean = y.mean(),std = y.std())}',
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                LambdaModule,
                'lambda x, y : { "Predictions" : dict(min = x.min(),max = x.max(),mean = x.mean(),std = x.std()),'
                                '"Targets"     : dict(min = y.min(),max = y.max(),mean = y.mean(),std = y.std())}',
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Stats',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torch.nn.BCEWithLogitsLoss,
                target_name     = 'Segmentor',
            ),
            **object_to_config(
                torch.nn.CrossEntropyLoss,
                target_name     = 'Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'CrossEntropyLoss',
        ),
        **object_to_config(
            MultiHeadLoss,
            Segmentor = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torch.nn.MSELoss,
                ),
                pred_apply  = 'lambda pred: torch.sigmoid(pred)',
            ),
            Classifier = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torch.nn.MSELoss,
                ),
                pred_apply  = 'lambda pred: torch.softmax(pred,-1)',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'MSE',
        ),
        **object_to_config(
            MultiHeadLoss,
            Segmentor = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torch.nn.L1Loss,
                ),
                pred_apply  = 'lambda pred: torch.sigmoid(pred)',
            ),
            Classifier = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torch.nn.L1Loss,
                ),
                pred_apply  = 'lambda pred: torch.softmax(pred,-1)',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'MAE',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Accuracy,
                task                ='multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(labels),
                target_name         ='Classifier',
            ),
            reduction       = 'sum',
            expect_type     = 'dict',
            target_name     = 'Accuracy -- Global',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Accuracy,
                task                ='multilabel',
                multidim_average    ='global',
                average             ='none',
                num_labels          = len(labels),
                target_name         ='Classifier',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'Accuracy -- Label-wise',
        ),
        **object_to_config(
            MultiHeadLoss,
            **object_to_config(
                torchmetrics.Recall,
                task                ='multilabel',
                multidim_average    ='global',
                average             ='micro',
                num_labels          = len(labels),
                target_name         ='Classifier',
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
                num_labels          = len(labels),
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
                num_labels          = len(labels),
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
                num_labels          = len(labels),
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
                num_labels          = len(labels),
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
                num_labels          = len(labels),
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
            Segmentor = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torchmetrics.image.StructuralSimilarityIndexMeasure,
                    data_range      = 1.0,
                    # **object_to_config(                               # In order to handle images smaller than 160x160
                    #     tuple,
                    #     [0.0448, 0.2856, 0.3001],
                    #     target_name = 'betas'
                    # ),
                    reduction       = 'elementwise_mean',
                ),
                pred_apply  = f'lambda target: target.reshape(target.shape[0],-1,int(target.shape[-1]**0.5),int(target.shape[-1]**0.5),)',
                targ_apply  = f'lambda target: target.reshape(target.shape[0],-1,int(target.shape[-1]**0.5),int(target.shape[-1]**0.5),)',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'SSIM',
        ),
        **object_to_config(
            MultiHeadLoss,
            Classifier = object_to_config(
                ProcessAndApplyMetric,
                object_to_config(
                    torchmetrics.AUROC,
                    task            ='multiclass',
                    num_classes     = len(labels),
                    thresholds      = 100,
                ),
                targ_apply  = 'lambda target: target.long().argmax(-1)',
            ),
            reduction       = 'none',
            expect_type     = 'dict',
            target_name     = 'AUROC',
        ),
    }
    pdir, train_config['hash'] = build_training_dir(
        train_config, 
        top_dir         = args.dest_top_dir, 
        test_version    = args.test_version
    )
    if not args.export :
        print(f'Test directory : {pdir}')

    if args.hash:
        print(train_config['hash'])

    if args.export :
        os.makedirs(os.path.dirname(pdir), exist_ok=True)
        pdir = save_config(train_config, pdir)
        
    if not args.hash:
        print(os.path.dirname(pdir))
