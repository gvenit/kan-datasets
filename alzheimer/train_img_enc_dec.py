#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

parser = ArgumentParser(
    description='Training script for the Ship Performance Clusterring Dataset.'
)

parser.add_argument('-t', '--train-config', dest='train_config', help='The hash of the training configuration file.')
parser.add_argument('-m', '--model-config', dest='model_config', help='The hash of the model configuration file.')
parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
parser.add_argument('--test-version', dest='test_version', type=str, default=None)

args = parser.parse_args()

# Check argument validity
if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir, exist_ok=True)
    
else :
    raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

if args.model_config is None :
    raise ValueError(f'Cannot locate model configuration file.')
    
else:
    args.model_config = os.path.join('model_config', 'img_enc_dec', args.model_config)
    args.model_config = f'{args.model_config}.json'
    
    if not os.path.exists(args.model_config) and not os.path.isabs(args.model_config):
        args.model_config = os.path.join(args.test_dir, args.model_config)
        
    if not os.path.exists(args.model_config) :
        raise ValueError(f'Cannot locate model configuration file in specified location; got {args.model_config}')
    else :
        print(f'-- Using model configuration path "{args.model_config}"')
    
if args.train_config is None :
    raise ValueError(f'Cannot locate training configuration file.')
    
else:
    args.train_config = os.path.join('train_config', 'img_enc_dec', args.train_config)
    
    if args.test_version is not None:
        args.train_config = os.path.join(args.train_config, args.test_version)
    
    args.train_config = f'{args.train_config}.json'
    
    if not os.path.exists(args.train_config) and not os.path.isabs(args.train_config):
        args.train_config = os.path.join(args.test_dir, args.train_config)
        
    if not os.path.exists(args.train_config) :
        raise ValueError(f'Cannot locate model configuration file in specified location; got {args.train_config}')
    else :
        print(f'-- Using model configuration path "{args.train_config}"')
        
# import set_environment
    
import torch
from torch.utils.data import DataLoader

from kan_utils.config import *
from kan_utils.dataset import smart_split_indices
from kan_utils.training import train
from kan_utils.utils import set_seed, load_model, save_model

from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset, get_groups
from custom_dataset import AlzheimerDataset
import extract_statistics
from collections import OrderedDict

device = torch.device(
    # 'cpu'
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Check configuration file validity
train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
model_config = load_config(args.model_config)

# Instantiate models
img_enc = instantiate(model_config,'img_enc')
img_dec = instantiate(model_config,'img_dec')
model   = torch.nn.Sequential(OrderedDict([('img_enc', img_enc), ('img_dec', img_dec)]))
print('-- Model :', model)
img_enc.to(device)

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

dataset = AlzheimerDataset(
    normalize_dataset(expand_df_labels(build_dataset())), 
    input_cols      = model_config['input'],
    output_cols     = model_config['output'],
    input_img_dims  = model_config['input_img_dim'],
    output_img_dims = model_config['output_img_dim'],
    return_key      = False, 
    path_col        = 'Path',
    orientation     = 'fixed',
)
set_seed(train_config['seed'])
train_idx, val_idx, *_ = smart_split_indices(
    splits          = train_config['splits'],
    groups          = get_groups(),
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

# For re-raising errors
e_0 = None
try: 
    train(
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
except Exception as e:
    # Raise at the end of the "finally" block
    e_0 = e
finally:
    # Split best model state dict to separate files
    pth         = os.path.join(args.test_dir, 'models', '{epoch}')
    img_enc_pth = os.path.join(args.test_dir, 'img_enc', '{epoch}')
    img_dec_pth = os.path.join(args.test_dir, 'img_dec', '{epoch}')
    
    print(pth)

    for epoch in ('best', 'last'):
        if os.path.exists(pth.format(epoch=epoch)):
            load_model(model, pth.format(epoch=epoch))
            save_model(model.modules['img_enc'], img_enc_pth.format(epoch=epoch))
            save_model(model.modules['img_dec'], img_dec_pth.format(epoch=epoch))
            os.remove(pth.format(epoch=epoch))
            
    os.removedirs(os.path.dirname(pth))

    # Move history directory
    if os.path.exists(os.path.join(args.test_dir, 'history.json')):
        os.renames(
            os.path.join(args.test_dir, 'history.json'),
            os.path.join(args.test_dir, 'train_1_history.json'),
        )
    
    # Re-raise errors if any 
    if e_0 is not None:
        raise e_0