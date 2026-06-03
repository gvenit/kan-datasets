#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Training script for the CIFAR-100 Dataset.')
    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--seed', dest='seed', type=int, default=42)
    parser.add_argument('--input-height', '--height', dest='height', type=int, default=32)
    parser.add_argument('--input-width', '--width', dest='width', type=int, default=32)
    parser.add_argument('--input-channels', '--channels', dest='channels', type=int, default=3)
    parser.add_argument('--output-channels','--output_channels', dest='output_channels', type=int, default=1)
    parser.add_argument('--attention-size','--attention_size','--attention', dest='attention', type=int, default=4)
    parser.add_argument('--hidden', '--hidden-state', dest='hidden_state', action='extend', nargs="+")
    parser.add_argument('--encoded', '--encoded-state', dest='encoded_state', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='custom')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
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
    
    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.callbacks import *
    from kan_utils.models import *
    from kan_utils.utils import uses_momentum
    
    from prepare_dataset import get_class_names
    import custom_model

    # CIFAR-100: Calculate input features based on image size
    if args.resize is not None:
        h, w = args.resize
        num_features = h * w * 3  # 3 channels (RGB)
    else:
        num_features = 32*32*3
    features = [f'pixel_{i}' for i in range(num_features)]
    
    # CIFAR-100: 100 class labels
    labels = get_class_names()
    model_config = {}
    model_config['input']  = features
    model_config['output'] = labels
    
    model_config['input_img_dim']  = [args.channels, args.height, args.width]
    model_config['encoded_state']  = args.encoded_state[-1]
    
    args.hidden_state = [
        model_config['input_img_dim'].copy(),
        *[
            [_, _, _] for _ in args.hidden_state
        ]
    ]
    
    model_config.update(
        object_to_config(
            custom_model.CustomModel,
            # custom_model.AsymmetricImageKANEncoder,
            # input_shape         = model_config['input_img_dim'][1:],
            # hidden_state_size   = args.hidden_state.copy(),
            hidden_shapes       = args.hidden_state.copy(),
            encoded_state_size  = args.encoded_state.copy(),
            num_grids           = args.num_grids,
            grid_min            = args.grid_min,
            grid_max            = args.grid_max,
            inv_denominator     = args.scale,
            mode                = args.mode,
            residual            = args.residual,
            dynamic             = args.dynamic,
            normalize           = args.normalize,
            dropout_rate        = args.dropout,
            # reduce_dim          = -3,
            # reduction           ='sum',
            target_name         ='model',
    ))
    model_config['outputs_logits'] = True
    model_config['flatten'] = False
    train_config = get_default_training_config()
    train_config['task'] = 'multiclass'
    # train_config['sampler'] = ['Label']
    # train_config['sample_weight'] = 'Weight'
    train_config['splits'] = [0.8,0.2]  
    train_config.update(
        # object_to_config(
        #     CombinedLoss,
        #     object_to_config(
        #         WeightedLoss,
                object_to_config(
                    torch.nn.CrossEntropyLoss,
                    label_smoothing = 0.1,
            #     ),
            # ),
            # object_to_config(
            #     ProcessAndApplyMetric,
            #         object_to_config(
            #             WeightedLoss,
            #             torch.nn.MSELoss,
            #         ),
                # object_to_config(
                #     WeightedLoss,
                #     torch.nn.BCEWithLogitsLoss,
                # ),
            #     targ_apply  = f'lambda targ: torch.nn.functional.one_hot(targ.long(), {len(labels)}).float().squeeze(-2)',
            #     pred_apply  = object_to_config(
            #                     torch.nn.Softmax,
            #                     dim = -1
            #                 ),
            # ),
            target_name = 'criterion',
    ))
    train_config['epochs'] = args.epochs
    train_config['patience'] = args.patience
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
    train_config['probability'] = 0.25
    if args.resize is not None:
        train_config['resize'] = tuple(args.resize)
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
        #         torchmetrics.PrecisionRecallCurve,
        #         task            = 'multiclass',
        #         num_classes     = len(labels),
        #         thresholds      = 100,
        #     ),
        #     targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
        #     target_name = 'PrecisionRecallCurve',
        # ),
        **object_to_config(
            ProcessAndApplyMetric,
            object_to_config(
                torchmetrics.AUROC,
                task            = 'multiclass',
                num_classes     = len(labels),
                thresholds      = 100,
            ),
            targ_apply  = 'lambda target: target.to(torch.int64).squeeze(-1)',
            target_name     = 'AUROC',
        ),
    }
    train_config['callbacks_arguments'].update({
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            task        = train_config['task'],
            export_path = os.path.join(THIS_DIR,'dataset','tr_statistics.csv'),
            target_name = 'train_gatherer',
        ),
        **object_to_config(
            GatherStatistics,
            input_cols  = model_config['input'],
            output_cols = model_config['output'],
            task        = train_config['task'],
            export_path = os.path.join(THIS_DIR,'dataset','val_statistics.csv'),
            overwrite   = 1,
            target_name = 'val_gatherer',
        ),
    })
    train_config['callbacks']['epoch_start'].extend([
        # object_to_config(
        #     'lambda *args, epoch = 1, dataloader = None, **kwargs: (epoch % 15 == 1) and hasattr(dataloader.dataset,"allowed_labels") and '
        #     'dataloader.dataset.allowed_labels('
        #         f'torch.randint(0,{len(labels)},(5*(epoch//15 + 1),))'
        #     ')'
        # ),
        object_to_config(
            'lambda *args, dataloader = None, **kwargs: hasattr(dataloader.dataset,"shuffle") and '
            'dataloader.dataset.shuffle()'
        ),
    ])
    train_config['callbacks']['train_iter_start'].extend([
        # object_to_config(
        #     'lambda *args, model=None, iteration=0, epoch=0, epochs=1, dataloader=None, **kwargs : model._modules["kan"].dropout_rate.set('
        #         f'{args.dropout} * torch.sigmoid( torch.tensor( ((epoch + (iteration / len(dataloader)) - {int(args.epochs) / 2}) / {int(args.epochs) / 4}) )).item()'
        #     ')'
        # ),
        # object_to_config(
        #     'lambda *args, train_gatherer=None, **kwargs: train_gatherer(*args,**kwargs)',
        # ),
    ])
    # train_config['callbacks']['train_end'].extend([
    #     object_to_config(
    #         'lambda *args, train_gatherer=None, **kwargs: train_gatherer.finalize(*args,**kwargs)',
    #     ),
    # ])
    # train_config['callbacks']['eval_iter_start'].extend([
    #     object_to_config(
    #         'lambda *args, val_gatherer=None, **kwargs: val_gatherer(*args,**kwargs)',
    #     ),
    # ])
    # train_config['callbacks']['eval_metrics_start'].extend([
    #     object_to_config(
    #         'lambda *args, val_gatherer=None, **kwargs: val_gatherer.finalize(*args,**kwargs)',
    #     ),
    # ])
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
