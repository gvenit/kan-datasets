#!/usr/bin/env python3   
import sys, os
import json
from argparse import ArgumentParser
from prepare_dataset import get_prepared_dataset, NUMERIC_COLS, QUALITY_SCORE_COLS, CATEGORY_COLS, STATS_PATH

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Configuration script for the Coffee Quality Dataset.'
)

INPUT_COLS = NUMERIC_COLS.copy() + CATEGORY_COLS.copy()
OUTPUT_COLS = QUALITY_SCORE_COLS.copy() 

# INPUT_COLS = QUALITY_SCORE_COLS.copy() 
# OUTPUT_COLS = NUMERIC_COLS.copy() + CATEGORY_COLS.copy()

parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
parser.add_argument('--treat-quality-as-categorical', dest='treat_quality_as_categorical', action='store_true', 
                    help='Treat quality scores as categorical (one-hot encoded) instead of continuous values')
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
from prepare_dataset import set_df_labels, create_labels
from extract_statistics import extract_correlate
from custom_callbacks import *

model_config = get_default_model_config()

# Get the fully prepared dataset (cleaned, labeled, expanded, and normalized)
df = get_prepared_dataset(treat_quality_as_categorical=args.treat_quality_as_categorical)

# Get actual column names from the normalization stats file
# This ensures we match exactly what exists in the cached normalized dataset
if os.path.exists(STATS_PATH):
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    all_cols = list(stats.keys())
    # Filter to only columns that exist in the dataframe
    all_cols = [col for col in stats.keys() if col in df.columns]
    
    # Separate into input and output based on original column definitions
    input_cols = []
    output_cols = []
    
    for col in all_cols:
        # Check if this column belongs to an output category
        is_output = False
        for output_name in OUTPUT_COLS:
            # Exact match (for non-categorical columns)
            if col == output_name:
                is_output = True
                break
            # Categorical match - must start with output_name followed by _Is_
            # But NOT if there's a dot before _Is_ (e.g., Certification.Body_Is_X should not match Body)
            if col.startswith(f'{output_name}_Is_'):
                is_output = True
                break
        
        if is_output:
            output_cols.append(col)
        else:
            input_cols.append(col)
else:
    raise FileNotFoundError(f'Normalization stats file not found at {STATS_PATH}. Please run dataset preparation first.')

model_config['input']  = input_cols
model_config['output'] = output_cols
model_config['treat_quality_as_categorical'] = args.treat_quality_as_categorical
model_config.update(
    object_to_config(
        model_config['model'],
        target_name       = 'model',
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
    )
)
categories = pd.unique(pd.Series(df.columns).apply(lambda row: row[:row.find('_Is_')] if '_Is_' in row else None))
categories = [cat for cat in categories if cat is not None]
categories = [[
    label for label in df.columns
        if label.startswith(f'{category}_Is_')
    ]
        for category in categories
]
categories = [_ for _ in categories if len(_)]

# Separate categories for input and output
input_categories = [cat for cat in categories if any(col in input_cols for col in cat)]
output_categories = [cat for cat in categories if any(col in output_cols for col in cat)]

# Save categories to model config for use in callbacks and testing
model_config['input_categories'] = input_categories
model_config['output_categories'] = output_categories

train_config = get_default_training_config()
train_config.update(
    object_to_config(
        MixedLoss,
        target_name     = 'criterion',
        output_cols     = model_config['output'],
        categories      = output_categories,
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
        output_cols     = model_config['output'],
        categories      = output_categories,
        categoriesLoss  = torch.nn.BCEWithLogitsLoss,
        regressionLoss  = torch.nn.HuberLoss, # TestLoss,
        reduction       = 'mean',
    ),
    **object_to_config(
        MixedLoss,
        target_name     = 'Accuracy',
        output_cols     = model_config['output'],
        categories      = output_categories,
        **object_to_config(
            OneHotMulticlassAccuracy, 
            target_name = 'categoriesLoss', 
            average = 'micro' # NOTE: We want per-category accuracies so we should replace with 'none'
        ),
        **object_to_config(
            torchmetrics.R2Score, 
            target_name = 'regressionLoss', 
            multioutput = 'raw_values'
        ),
        reduction       = 'none',
    )
}
mask = object_to_config(
    MaskInput,
    input = model_config['input'],
    input_categories = input_categories,
    max_probability = 0.4,
    x_shift = 300 / int(train_config['epochs']),
    masked_value = -1,
)
train_config['callbacks']['epoch_start'].append(
    object_to_config(
        'lambda *args, probability_adjuster=None, criterion=None, **kwargs : criterion.update_probabilities(1-0.5*probability_adjuster.get_output_prob())'
    )
)
train_config['callbacks']['train_iter_start'].append(mask)
train_config['callbacks']['eval_iter_start'].append(mask)
train_config['callbacks']['epoch_end'].append(
    object_to_config(
        'lambda *args, probability_adjuster=None,**kwargs : probability_adjuster(*args,**kwargs)'
    )
)
train_config['callbacks']['training_finished'].append(
    object_to_config(
        'lambda *args, probability_adjuster=None,**kwargs : probability_adjuster.save_logs()'
    )
)

def build_test_dir(train_config, model_config, top_dir = None, test_version = None):
    pdir = os.path.join(
        '_'.join([ str(_) for _ in model_config['model_kwargs']['hidden_layers']]),
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

# Compute correlation matrix from the expanded dataframe (with one-hot encoded columns)
# This ensures column names match what the model actually uses
df_corr = extract_correlate(df)
  
train_config['callbacks_arguments'].update( object_to_config(
    ProbabilityAdjuster,
    target_name='probability_adjuster',
    input               = model_config['input'],
    input_categories    = input_categories,
    output              = model_config['output'],
    output_categories   = output_categories,
    confusion_matrix    = df_corr.to_dict(),
    smoothing_coef      = 0.1,
    saving_interval     = 25,
    log_dir             = pdir,
))
  
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
    