#!/usr/bin/env python3

import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Training script for the Ship Performance Clusterring Dataset.'
)

parser.add_argument('-t', '--train-config', dest='train_config', help='The path to the JSON training configuration file.')
parser.add_argument('-m', '--model-config', dest='model_config', help='The path to the JSON model configuration file.')
parser.add_argument('-d', '--test-dir', dest='test_dir', default=THIS_DIR, help='The directory to be used as a top directory for training.')

args = parser.parse_args()

# Check argument validity
if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir, exist_ok=True)
    
else :
    raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

if args.train_config is None :
    path = os.path.join(args.test_dir, 'config', 'train.json')
    
    if os.path.exists(path):
        args.train_config = path
        print(f'-- Using default training configuration path "{path}"')
        
    else :
        raise ValueError(f'Cannot locate training configuration file.')
    
elif not os.path.exists(args.train_config) :
    if os.path.isabs(args.train_config):
        raise ValueError(f'Cannot locate training configuration file in specified location; got {args.train_config}')
        
    else :
        path = os.path.join(args.test_dir, args.train_config)
        
        if os.path.exists(path):
            args.train_config = path
            print(f'-- Using training configuration path "{path}"')
            
        else :
            raise ValueError(f'Cannot locate training configuration file.')
        
if args.model_config is None :
    path = os.path.join(args.test_dir, 'config', 'model.json')
    
    if os.path.exists(path):
        args.model_config = path
        print(f'-- Using default model configuration path "{path}"')
        
    else :
        raise ValueError(f'Cannot locate model configuration file.')
    
elif not os.path.exists(args.model_config) :
    if os.path.isabs(args.model_config) :
        raise ValueError(f'Cannot locate model configuration file in specified location; got {args.model_config}')
        
    else :
        path = os.path.join(args.test_dir, args.model_config)
        
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
            
        else :
            raise ValueError(f'Cannot locate model configuration file.')
        
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from kan_utils.utils import load_dict
import kan_utils.plotter as plotter

# from kan_utils.config import *
# from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset
import matplotlib.pyplot as plt

# Check configuration file validity
# train_config = load_config(args.train_config)
# model_config = load_config(args.model_config)

# Read training history
history = load_dict(os.path.join(args.test_dir, 'history'))

# Extract result statistics
plots_path = os.path.join(args.test_dir,'plot')
os.makedirs(plots_path, exist_ok=True)

## Training vs Validation Loss
epochs   = np.asarray(list(history['train'].keys()), dtype=int)
tr_loss  = [_['loss'] for _ in history['train'].values()]
# [print(_) for _ in history['val'].values()]
val_loss = [_['loss'] for _ in history['val'].values()]

plt.plot(epochs, tr_loss, val_loss)

plt.title('Training vs Validation Loss')
plt.legend(['training','validation'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

save_path = os.path.join(plots_path, 'tr_vs_val.png')
plt.savefig(save_path)
plt.close('all')
print(f"Training vs Validation diagram saved to: {save_path}")

# Read ground truth and predictied values
rslt_path = os.path.join(args.test_dir,'rslt')

gt_df = pd.read_csv(os.path.join(rslt_path, 'ground_truth.csv'), index_col='Index')
pr_df = pd.read_csv(os.path.join(rslt_path, 'best.csv'), index_col='Index')

# Read Categories
categories = load_dict(os.path.join(THIS_DIR, 'dataset', 'labels'))

# Extract Confusion Matrices for each set of categories
categorical_cols = []
for category, types in categories.items() :
    class_names = list(types.keys())
    
    # Find columns of the specified category
    cols = [col for col in gt_df.columns if category in col]
    categorical_cols.extend(cols)
    
    # Get DataFrame slices
    gt_slice = gt_df[cols].copy()
    pr_slice = pr_df[cols].copy()
    
    
    # Fix DataFrames
    gt_slice.columns = [col[col.find('_Is_')+4:] for col in gt_slice.columns]
    pr_slice.columns = [col[col.find('_Is_')+4:] for col in pr_slice.columns]
    
    # Get probabilities with Softmax
    pr_slice = pr_slice.loc[pr_slice.index].apply(np.exp)
    pr_slice = pr_slice.loc[pr_slice.index].apply(
        (lambda row : row / row.sum()),
        axis=1
    )
    
    # Find probabilities of the unknown class
    gt_type = gt_slice.apply(
        lambda row: gt_slice.columns[np.argmax(row)],
        axis=1
    )
    pr_type = pr_slice.apply(
        lambda row: pr_slice.columns[np.argmax(row)],
        axis=1
    )
    cm = confusion_matrix(gt_type.values, pr_type.values, labels=class_names)
    save_path = os.path.join(plots_path, f'cm_{category}.png')
    plotter.plot_confusion_matrix(
        cm, 
        class_names, 
        normalize = True, 
        title     = f'Confusion Matrix : {category}',
        save_path = save_path
    )
    print(f"{category} Confusion matrix saved to: {save_path}")
    plt.close('all')

# Plot Reggression-type Columns
idx = gt_df.index.values
for col in gt_df.columns[np.isin(gt_df.columns, categorical_cols, invert=True)]:
    plt.plot(idx, gt_df[col], pr_df[col])

    plt.title(col)
    plt.legend(['Ground Truth','Prediction'])
    plt.xlabel('Index')
    plt.ylabel(col)

    save_path = os.path.join(plots_path, f'{col}.png')
    plt.savefig(save_path)
    print(f"{col} Diagram saved to: {save_path}")
    plt.close('all')