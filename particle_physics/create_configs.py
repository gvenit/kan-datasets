#!/usr/bin/env python3   
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
parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
parser.add_argument('--residual', dest='residual', action='store_true')
parser.add_argument('--dynamic', dest='dynamic', action='store_true')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
parser.add_argument('--patience', dest='patience', default=10)
parser.add_argument('--epochs', dest='epochs', default=500)
parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
parser.add_argument('--test-version', dest='test_version', type=str, default='0')

args = parser.parse_args()

import pandas as pd
import torch
import torchmetrics
import hashlib
from collections import OrderedDict

from kan_utils.config import *
from kan_utils.metrics import *
from kan_utils.callbacks import *
from kan_utils.models import LambdaModule, FasterKAN, RSWAFF
from kan_utils.utils import uses_momentum

from prepare_dataset import build_dataset, expand_df_labels
from extract_statistics import get_corellate
# from custom_callbacks import UpdatableFloat

model_config = {}
df : pd.DataFrame = expand_df_labels(build_dataset())
label_corr = get_corellate().loc['Label'].drop(['Label','Weight'])
# label_corr = label_corr.index[label_corr.abs() > 0.05].tolist()
label_corr = label_corr.index.tolist()

model_config['input']  = label_corr
model_config['output'] = ['Label',]
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
                        len(model_config['output']),
                    ],
                    num_grids         = args.num_grids,
                    grid_min          = args.grid_min,
                    grid_max          = args.grid_max,
                    inv_denominator   = args.scale,
                    mode              = args.mode,
                    residual          = args.residual,
                    dynamic           = args.dynamic,
                    **object_to_config(
                        UpdatableFloat,
                        0,
                        target_name   = 'dropout_rate'
                    )
                ),
            ],[
                'actf',
                torch.nn.Sigmoid,
                # RSWAFF,
                # object_to_config(
                #     LambdaModule,
                #     'lambda x: 1. - torch.nn.functional.tanh(x)**2'
                # ),
            ]
        ]),
        target_name       = 'model',
))
categories = pd.unique(pd.Series(df.columns).apply(lambda row: row[:row.find('_Is_')]))
categories = [[
    label for label in df.columns
        if f'{category}_Is_' in label
    ]
        for category in categories
]
categories = [_ for _ in categories if len(_)]

train_config = get_default_training_config()
train_config['sampler'] = ['Label']
# train_config['sample_weight'] = 'Weight'
# train_config['splits'] = [0.66,0.09,0.25]  
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
# mask = object_to_config(
#     MaskInput,
#     input            = model_config['input'],
#     input_categories = categories,
#     max_probability  = 0.4,
#     x_shift          = 300 / int(train_config['epochs']),
#     masked_value     = 0.5,
# )
# train_config['callbacks']['epoch_start'].append(
#     object_to_config(
#         'lambda *args, probability_adjuster=None, criterion=None, **kwargs : criterion.update_probabilities(1-0.5*probability_adjuster.get_output_prob())'
#     )
# )
train_config['callbacks']['train_iter_start'].append(
    object_to_config(
        'lambda *args, model=None, iteration=0, epoch=0, epochs=1, dataloader=None, **kwargs : model._modules["kan"].dropout_rate.set('
            f'{args.dropout} * torch.sigmoid( torch.tensor( ((epoch + (iteration / len(dataloader)) - {int(args.epochs) / 2}) / {int(args.epochs) / 4}) )).item()'
        ')'
    )
)
# train_config['callbacks']['train_iter_start'].append(mask)
# train_config['callbacks']['eval_iter_start'].append(mask)
# train_config['callbacks']['epoch_end'].append(
#     object_to_config(
#         'lambda *args, probability_adjuster=None,**kwargs : probability_adjuster(*args,**kwargs)'
#     )
# )
# train_config['callbacks']['training_finished'].append(
#     object_to_config(
#         'lambda *args, probability_adjuster=None,**kwargs : probability_adjuster.save_logs()'
#     )
# )
# train_config['callbacks_arguments'].update( object_to_config(
#     ProbabilityAdjuster,
#     target_name         ='probability_adjuster',
#     input               = model_config['input'],
#     input_categories    = categories,
#     output              = model_config['output'],
#     output_categories   = categories,
#     confusion_matrix    = get_corellate().to_dict(),
#     smoothing_coef      = 0.1,
#     saving_interval     = 25,
#     # log_dir             = pdir,
# ))
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

# train_config['callbacks_arguments'].update( object_to_config(
#     ProbabilityAdjuster,
#     target_name         ='probability_adjuster',
#     input               = model_config['input'],
#     input_categories    = categories,
#     output              = model_config['output'],
#     output_categories   = categories,
#     confusion_matrix    = get_corellate().to_dict(),
#     smoothing_coef      = 0.1,
#     saving_interval     = 25,
#     log_dir             = pdir,
# ))
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
