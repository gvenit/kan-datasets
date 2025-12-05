#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Training script for the Ship Performance Clusterring Dataset.'
)

parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
parser.add_argument('--test-version', dest='test_version', default=None)
parser.add_argument('--seed', dest='seed', type=int, default=42)
parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
parser.add_argument('--residual', dest='residual', action='store_true')
parser.add_argument('--patience', dest='patience', default=100)
parser.add_argument('--epochs', dest='epochs', default=500)
parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--export', action='store_true',dest='export')

args = parser.parse_args()

import pandas as pd
import torch
import torchmetrics

from kan_utils.config import *
from kan_utils.metrics import *
from prepare_dataset import build_datset, expand_df_labels
from custom_callbacks import MaskInput

model_config = get_default_model_config()

df = expand_df_labels(build_datset())

model_config['input']  = df.columns
model_config['output'] = df.columns
model_config.update(
    object_to_config(
        model_config['model'],
        target_name       = 'model',
        layers_hidden     = [
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
    )
)
categories = pd.unique(pd.Series(df.columns).apply(lambda row: row[:row.find('_Is_')]))
categories = [[
    label for label in df.columns
        if f'{category}_Is_' in label
    ]
        for category in categories
]
categories = [_ for _ in categories if len(_)]

train_config = get_default_training_config()
model_config.update(
    object_to_config(
        MixedLoss,
        target_name     = 'criterion',
        output_cols     = df.columns.tolist(),
        categories      = categories,
        categoriesLoss  = torch.nn.BCEWithLogitsLoss,
        regressionLoss  = torch.nn.HuberLoss, # TestLoss,
        reduction       = 'random',
))
train_config['patience'] = args.patience
train_config['epochs'] = args.epochs
train_config.update(
    object_to_config(
        getattr(torch.optim, args.optimizer),
        target_name     = 'optimizer',
        weight_decay    = args.weight_decay,
        **({
            'momentum' : args.momentum
        } if args.optimizer in ('SGD', 'RMSprop') else {})
))
train_config['lr'] = args.lr
train_config['seed'] = args.seed
train_config['batch_size'] = args.batch_size
train_config['eval_criteria'] = {
    **object_to_config(
        MixedLoss,
        target_name     = 'loss',
        output_cols     = df.columns.tolist(),
        categories      = categories,
        categoriesLoss  = torch.nn.BCEWithLogitsLoss,
        regressionLoss  = torch.nn.HuberLoss, # TestLoss,
        reduction       = 'sum',
    ),
    **object_to_config(
        MixedLoss,
        target_name     = 'Accuracy',
        output_cols     = df.columns.tolist(),
        categories      = categories,
        **object_to_config(
            OneHotMulticlassAccuracy, 
            target_name = 'categoriesLoss', 
            average = 'micro'
        ),
        regressionLoss  = torchmetrics.R2Score, # TestLoss,
        reduction       = 'none',
    )
}
mask = object_to_config(
    MaskInput,
    input = model_config['input'],
    input_categories = categories,
    max_probability = 0.4,
    x_shift = 300 / int(train_config['epochs']),
    masked_value = -1,
)
train_config['callbacks']['train_iter_start'].append(mask)
train_config['callbacks']['eval_iter_start'].append(mask)

def build_test_dir(train_config, model_config, top_dir = None, test_version = None):
    pdir = os.path.join(
        '_'.join([ str(_) for _ in model_config['model_kwargs']['layers_hidden']]),
        '_'.join([
            'm', *[str(_) for _ in model_config['model_kwargs']['grid_min']],
            'M', *[str(_) for _ in model_config['model_kwargs']['grid_max']],
            's', *[str(_) for _ in model_config['model_kwargs']['inv_denominator']],
        ]),
        find_class_name(train_config['scheduler']),
        '_'.join([
            find_class_name(train_config['optimizer']),
            'lr',
            str(train_config['lr'])
        ]),
        find_class_name(train_config['criterion']),
        '_'.join(['seed', str(train_config['seed'])]),
    )
    if top_dir is not None:
        pdir = os.path.join(top_dir,pdir)
    if test_version is not None:
        pdir = os.path.join(pdir,'_'.join(['test',test_version]))
    return pdir

pdir = build_test_dir(train_config, model_config, top_dir=args.dest_top_dir, test_version=args.test_version)
if not args.export :
    print(f'Test directory : {pdir}')
    
path = os.path.join(pdir,'config','train.json')
save_config(train_config, path)
if not args.export :
    print(f'Training configuration saved in "{path}".')

path = os.path.join(pdir,'config','model.json')
save_config(model_config, path)
if not args.export :
    print(f'Training configuration saved in "{path}".')

if args.export :
    print(pdir)
    