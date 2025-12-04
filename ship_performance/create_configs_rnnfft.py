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
parser.add_argument('--fft', dest='fft', type=str, default='OptimisedRNNFFT')
parser.add_argument('--test-version', dest='test_version', default=None)
parser.add_argument('--seed', dest='seed', type=int, default=42)
parser.add_argument('--layers', '--hidden-layers', dest='layers', type=int, action='extend', nargs="+")
parser.add_argument('--radix', dest='radix', action='extend', type=int, nargs="?", default=[2,])
parser.add_argument('--epochs', dest='epochs', default=500)
parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=16)
parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--export', action='store_true',dest='export')

args = parser.parse_args()

import pandas as pd
import numpy as np
import torch
import torchmetrics

from kan_utils.config import find_class_name, get_default_model_config, get_default_training_config, save_config
from kan_utils.metrics import MixedLoss
from kan_utils.models import RNNFFT, OptimisedRNNFFT
from kan_utils.utils import expand_value
from kan_utils.config import find_object_from_name
from prepare_dataset import build_datset, expand_df_labels

model_config = get_default_model_config()

df = expand_df_labels(build_datset())

model_config['input']  = df.columns
model_config['output'] = df.columns

args.radix = expand_value(args.radix, len(args.layers))

num_features = len(model_config['input']), *(np.array(args.radix) ** np.array(args.layers)).astype('int').tolist(), len(model_config['output']),

model_config['model']  = torch.nn.Sequential
model_config.pop('model_kwargs')
model_config['model_args'] = []

model_config['model_args'].extend([
    {
        '' : torch.nn.Linear,
        '_kwargs' : {
            'in_features' : num_features[0],
            'out_features': num_features[1],
            'bias' : True,
        }
    },
    {'' : torch.nn.ReLU},
])
for layer, radix, _nfeatures, _nfeatures_next in zip(args.layers, args.radix, num_features[1:-1], num_features[2:]):
    model_config['model_args'].extend([
        {
            # '' : RNNFFT,
            '' : find_object_from_name(args.fft),
            '_kwargs' : {
                'input_size' : _nfeatures,
                'radix': radix,
                'residual' : 1,
            }
        },
        {'' : torch.nn.ReLU},
    ])
    if _nfeatures != _nfeatures_next:
        model_config['model_args'].extend([
            {
                '' : torch.nn.Linear,
                '_kwargs' : {
                    'in_features' : _nfeatures,
                    'out_features': _nfeatures_next,
                    'bias' : True,
                }
            },
            {'' : torch.nn.ReLU},
        ])

categories = pd.unique(pd.Series(df.columns).apply(lambda row: row[:row.find('_Is_')]))
categories = [[
    label for label in df.columns
        if f'{category}_Is_' in label
    ]
        for category in categories
]
categories = [_ for _ in categories if len(_)]

train_config = get_default_training_config()
train_config['criterion'] =  MixedLoss
train_config['criterion_kwargs'] = {
    'output_cols' : df.columns.tolist(),
    'categories'  : categories,
    'categoriesLoss' : torch.nn.BCEWithLogitsLoss,
    'regressionLoss' : torch.nn.HuberLoss,
}
train_config['epochs'] = args.epochs
train_config['optimizer'] = getattr(torch.optim, args.optimizer)
train_config['optimizer_kwargs'] = {
    'weight_decay' : args.weight_decay,
    **({
        'momentum' : args.momentum
    } if args.optimizer in ('SGD', 'RMSprop') else {})
}
train_config['lr'] = args.lr
train_config['seed'] = args.seed
train_config['batch_size'] = args.batch_size

train_config['eval_criteria'].update({
    'loss' : MixedLoss,
    'loss_kwargs' : {
        'output_cols'    : df.columns.tolist(),
        'categories'     : categories,
        'categoriesLoss' : torch.nn.BCEWithLogitsLoss,
        'regressionLoss' : torch.nn.HuberLoss, # TestLoss,
        'reduction'      : 'sum'
    },
    'Accuracy' : MixedLoss,
    'Accuracy_kwargs' : {
        'output_cols'    : df.columns.tolist(),
        'categories'     : categories,
        'categoriesLoss' : torchmetrics.classification.MulticlassAccuracy,
        'regressionLoss' : torchmetrics.R2Score, # TestLoss,
        'reduction'      : 'none'
    },
})

def build_test_dir(train_config, fft, num_features, top_dir = None, test_version = None):
    pdir = os.path.join(
        fft,
        '_'.join([ str(_) for _ in num_features]),
        '_'.join([
            'r', '_'.join([ str(_) for _ in args.radix]),
            'd', '_'.join([ str(_) for _ in args.layers])
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

pdir = build_test_dir(train_config, args.fft, num_features, top_dir=args.dest_top_dir, test_version=args.test_version)
if not args.export :
    print(f'Test directory : {pdir}')
    
path = os.path.join(pdir,'config','model.json')
save_config(model_config, path)
if not args.export :
    print(f'Training configuration saved in "{path}".')

path = os.path.join(pdir,'config','train.json')
save_config(train_config, path)
if not args.export :
    print(f'Training configuration saved in "{path}".')

if args.export :
    print(pdir)
    