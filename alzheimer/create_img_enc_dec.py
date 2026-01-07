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
parser.add_argument('--input-height', '--input_height', dest='input_height', type=int, default=64)
parser.add_argument('--output-height', '--output_height', dest='output_height', type=int, default=64)
parser.add_argument('--input-width', '--input_width', dest='input_width', type=int, default=64)
parser.add_argument('--output-width', '--output_width', dest='output_width', type=int, default=64)
parser.add_argument('--input-depth', '--input_depth', dest='input_depth', type=int, default=64)
parser.add_argument('--output-depth','--output_depth', dest='output_depth', type=int, default=64)
parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
parser.add_argument('--scale','--inv_denominator', dest='inv_denominator', action='extend', nargs="+")
parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
parser.add_argument('--residual', dest='residual', action='store_true')
parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")

args = parser.parse_args()

import torch
import hashlib

from kan_utils.config import *
from kan_utils.metrics import *
from kan_utils.models import *
from kan_utils.utils import expand_value
# from custom_callbacks import *

model_config = {}
model_config['input_img_dim']  = [args.input_depth, args.input_height, args.input_width]
model_config['output_img_dim'] = [args.output_depth, args.output_height, args.output_width]
model_config['input']  = ['Path']
model_config['output'] = ['Path']

args.hidden_layers   = [args.input_width * args.input_height, *args.hidden_layers]
args.num_grids       = expand_value(args.num_grids,       len(args.hidden_layers)-1)
args.grid_min        = expand_value(args.grid_min,        len(args.hidden_layers)-1)
args.grid_max        = expand_value(args.grid_max,        len(args.hidden_layers)-1)
args.inv_denominator = expand_value(args.inv_denominator, len(args.hidden_layers)-1)

model_config.update(
    object_to_config(
        SubBatch,
        target_name         ='img_enc',
        input_data_dim      = -2,
        output_data_dim     = -1,
        **object_to_config(
            torch.nn.Sequential,
            torch.nn.Flatten,
            object_to_config(
                FasterKAN,
                hidden_layers   = args.hidden_layers.copy(),
                num_grids       = args.num_grids.copy(),
                grid_min        = args.grid_min.copy(),
                grid_max        = args.grid_max.copy(),
                inv_denominator = args.inv_denominator.copy(),
                mode            = args.mode,
                residual        = args.residual,
            ),
            target_name     ='model',
        ),
))

# Reverse order for decoder
args.hidden_layers.reverse()   
args.num_grids.reverse()       
args.grid_min.reverse()        
args.grid_max.reverse()        
args.inv_denominator.reverse() 

args.hidden_layers = args.hidden_layers[:-1] + [args.output_width * args.output_height]

model_config.update(
    object_to_config(
        SubBatch,
        target_name         ='img_dec',
        input_data_dim      = -1,
        output_data_dim     = -2,
        **object_to_config(
            torch.nn.Sequential,
            object_to_config(
                FasterKAN,
                hidden_layers   = args.hidden_layers.copy(),
                num_grids       = args.num_grids.copy(),
                grid_min        = args.grid_min.copy(),
                grid_max        = args.grid_max.copy(),
                inv_denominator = args.inv_denominator.copy(),
                mode            = args.mode,
                residual        = args.residual,
            ),
            torch.nn.Sigmoid,
            object_to_config(
                Reshaper,
                input_data_shape  = args.hidden_layers[-1:],
                output_data_shape = model_config['output_img_dim'][1:],
            ),
            target_name     ='model',
        ),
))

def build_img_enc_dec_dir(args, top_dir = None,):
    pdir = os.path.join(
        '_'.join([str(_) for _ in [args.input_width, args.input_height]]),
        '_'.join([str(_) for _ in [args.output_width, args.output_height]]),
        '_'.join([str(_) for _ in args.hidden_layers[:-1]]),
        '_'.join([str(_) for _ in args.num_grids]),
        '_'.join([
            'm', *[str(_) for _ in args.grid_min],
            'M', *[str(_) for _ in args.grid_max],
            's', *[str(_) for _ in args.inv_denominator],
        ]),
        args.mode,
        str(args.residual)
    )
    hashed = hashlib.sha1(pdir.encode()).hexdigest()
    pdir = os.path.join('model_config', 'img_enc_dec', hashed)
    if top_dir is not None:
        pdir = os.path.join(top_dir,pdir)
    return pdir, hashed

pdir, model_config['hash'] = build_img_enc_dec_dir(args, top_dir=args.dest_top_dir)

if args.hash:
    print(model_config['hash'])

if args.export :
    pdir = save_config(model_config, pdir)
    
if not args.hash:
    print(pdir)
