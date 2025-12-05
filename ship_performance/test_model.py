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
        
import set_environment
        
import pandas as pd
import torch
from torch.utils.data import DataLoader
from kan_utils.utils import load_model, load_dict, save_dict
from kan_utils.dataset import DataFrameToDataset, split_dataset
from kan_utils.performance import get_summary
from kan_utils.config import *
from kan_utils.training import evaluate
from prepare_dataset import build_datset, expand_df_labels, normalize_dataset
from extract_statistics import extract_statistics
import custom_callbacks


device = torch.device(
    # 'cpu'
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Check configuration file validity
train_config = load_config(args.train_config, locals={**custom_callbacks.__dict__})
model_config = load_config(args.model_config)

# Instantiate models
model = instantiate(model_config,'model')

# Load model state dict
fname = os.path.join(args.test_dir, 'models', 'best')
model = load_model(model, fname)

# Instantiate evaluation criteria
eval_criteria = {
    **weak_instantiate_all(train_config['eval_criteria'])
}
if 'loss' not in eval_criteria.keys():
    eval_criteria.update({
        'loss' : instantiate(train_config,'criterion'),
    })
print('-- Evaluation Criteria :')
if len(eval_criteria):
    for key, val in eval_criteria.items():
        print('  --', key, ':', val)
else :
    print('  No evaluation criteria.')

*_, test_loader = split_dataset(
    splits          = train_config['splits'],
    full_dataset    = DataFrameToDataset(
        normalize_dataset(expand_df_labels(build_datset())),
        input_cols  = model_config['input'],
        output_cols = model_config['output'],
        return_key  = True,
    ),
    seed            = train_config['seed']
)

print(
    '-- Model Summary :',
    get_summary(
        model,
        next(iter(test_loader))[0],
        dest = os.path.join(args.test_dir, 'models', 'summary')
    )
)

test_loader = DataLoader(
    test_loader, 
    shuffle         = False,
    batch_size      = train_config['batch_size'],
    num_workers     = os.cpu_count(),
    pin_memory      = device == torch.device('cuda'),
)

print('-- Using dataset split :', train_config['splits'])
print('  -- Test       :', len(test_loader.dataset))

test_metrics = evaluate(
    model,
    eval_dataloader   = test_loader,
    criteria          = eval_criteria,
    keep_copy         = True,
    checkpoint_path   = fname.replace('models','rslt'),
    epoch             = 'best',
    show_pbar         = True,
    device            = device,
)

hist_path = os.path.join(args.test_dir,'history')
history = load_dict(hist_path)

history['test'] = {
    'best' : test_metrics
}
save_dict(history, hist_path)

# Separate ground truth and predicted values
rslt_path = os.path.join(args.test_dir,'rslt')
test_df = pd.read_csv(os.path.join(rslt_path,'best.csv'), index_col='Index')

gt_df = test_df[[_ for _ in test_df.columns if 'targ' in _]]
pr_df = test_df[[_ for _ in test_df.columns if 'pred' in _]]

gt_df.columns = model_config['output']
pr_df.columns = model_config['output']

gt_df = normalize_dataset(gt_df, reverse=True)
pr_df = normalize_dataset(pr_df, reverse=True)

gt_df.to_csv(os.path.join(rslt_path, 'ground_truth.csv'))
pr_df.to_csv(os.path.join(rslt_path, 'best.csv'))
test_df.to_csv(os.path.join(rslt_path, 'rslt.csv'))

extract_statistics(pr_df, output_dir=rslt_path)