#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Training script for the Coffee Quality Dataset.'
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
        
import torch
from torch.utils.data import DataLoader
from kan_utils.dataset import DataFrameToDataset, split_dataset
from kan_utils.config import *
from kan_utils.training import train
from prepare_dataset import get_prepared_dataset
import extract_statistics
import custom_callbacks

device = torch.device(
    # 'cpu'
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Check configuration file validity
train_config = load_config(args.train_config, locals=get_locals(custom_callbacks, extract_statistics))
model_config = load_config(args.model_config)

# Instantiate models
model = instantiate(model_config,'model')
print('-- Model :', model)
model.to(device)

# Instantiate criterion
criterion = instantiate(train_config,'criterion')
print('-- Criterion :', criterion)

# Instantiate optimizer
optimizer = instantiate(train_config,'optimizer', model.parameters(), lr = train_config['lr'])
print('-- Optimizer :', optimizer)

# Instantiate scheduler
scheduler = instantiate(train_config,'scheduler', optimizer)
print('-- Scheduler :', scheduler)

# Instantiate evaluation criteria
eval_criteria = weak_instantiate_all(train_config['eval_criteria'])
print('-- Evaluation Criteria :')
if len(eval_criteria):
    for key, val in eval_criteria.items():
        print('  --', key, ':', val)
else :
    print('  No evaluation criteria.')
    
# Instantiate callbacks
callbacks = weak_instantiate_all(train_config['callbacks'])
callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

# print(callbacks_arguments)

# Load the prepared dataset using the treat_quality_as_categorical setting from config
treat_quality_as_categorical = model_config.get('treat_quality_as_categorical', False)
train_loader, val_loader, *_ = split_dataset(
    splits          = train_config['splits'],
    full_dataset    = DataFrameToDataset(
        get_prepared_dataset(treat_quality_as_categorical=treat_quality_as_categorical),
        input_cols  = model_config['input'],
        output_cols = model_config['output'],
    ),
    seed            = train_config['seed']
)

train_loader = DataLoader(
    train_loader, 
    shuffle             = True,
    batch_size          = train_config['batch_size'],
    num_workers         = os.cpu_count(),
    pin_memory          = device == torch.device('cuda'),
    persistent_workers  = True,
)
val_loader = DataLoader(
    val_loader, 
    shuffle             = False,
    batch_size          = train_config['batch_size'],
    num_workers         = os.cpu_count(),
    pin_memory          = device == torch.device('cuda'),
    persistent_workers  = True,
)

print('-- Using dataset split :', train_config['splits'])
print('  -- Train      :', len(train_loader.dataset))
print('  -- Validation :', len(val_loader.dataset))

history = train(
    model,
    train_dataloader    = train_loader,
    eval_dataloader     = val_loader,
    criterion           = criterion,
    eval_criteria       = eval_criteria,
    optimizer           = optimizer,
    scheduler           = scheduler,
    epochs              = train_config['epochs'],
    patience            = 100,
    top_dirname         = args.test_dir,
    device              = device,
    evaluate_training   = False,
    show_pbar           = 'external',
    callbacks           = callbacks,
    callbacks_arguments = callbacks_arguments,
)