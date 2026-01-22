#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__' :
    parser = ArgumentParser(
        description='Model configuration script for the Alzheimer\'s Dataset.'
    )

    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--input-height', '--input_height', dest='input_height', type=int, default=64)
    parser.add_argument('--output-height', '--output_height', dest='output_height', type=int, default=64)
    parser.add_argument('--input-width', '--input_width', dest='input_width', type=int, default=64)
    parser.add_argument('--output-width', '--output_width', dest='output_width', type=int, default=64)
    parser.add_argument('--input-depth', '--input_depth', dest='input_depth', type=int, default=64)
    parser.add_argument('--output-depth','--output_depth', dest='output_depth', type=int, default=64)
    parser.add_argument('--hidden', '--hidden-state', dest='hidden_state', action='extend', nargs="+")
    parser.add_argument('--encoded', '--encoded-state', dest='encoded_state', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', type=int, default=5)
    parser.add_argument('--grid-min', dest='grid_min', type=float, default=-1.)
    parser.add_argument('--grid-max', dest='grid_max', type=float, default= 1.)
    parser.add_argument('--scale','--inv_denominator', dest='inv_denominator', type=float, default= 1.5)
    parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    import torch

    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.models import *
    from kan_utils.utils import expand_value
    
    import custom_model
    # from custom_callbacks import *
    
    from build_model import build_train_1_model, build_model_dir

    args.test_version = '_'.join(['test',args.test_version])

    model_config = {}
    model_config['input_img_dim']  = [args.input_depth, args.input_height, args.input_width]
    model_config['output_img_dim'] = [args.output_depth, args.output_height, args.output_width]
    model_config['encoded_state']  = args.encoded_state[-1]
    model_config['input']  = ['Path']
    model_config['output'] = ['Path']

    model_config.update(
        object_to_config(
            SubBatch,
            target_name         ='img_enc',
            input_data_dim      = -2,
            output_data_dim     = -1,
            **object_to_config(
                custom_model.ImageKANEncoder,
                input_shape         = model_config['input_img_dim'][1:],
                hidden_state_size   = args.hidden_state.copy(),
                encoded_state_size  = args.encoded_state.copy(),
                num_grids           = args.num_grids,
                grid_min            = args.grid_min,
                grid_max            = args.grid_max,
                inv_denominator     = args.inv_denominator,
                mode                = args.mode,
                residual            = args.residual,
                dynamic             = args.dynamic,
                dropout_rate        = args.dropout,
                target_name         ='model',
            ),
    ))
    args.hidden_state.reverse()
    args.encoded_state.reverse()
    
    model_config.update(
        object_to_config(
            SubBatch,
            target_name         ='img_dec',
            input_data_dim      = -1,
            output_data_dim     = -2,
            **object_to_config(
                torch.nn.Sequential,
                object_to_config(
                    custom_model.ImageKANDecoder,
                    output_shape        = model_config['output_img_dim'][1:],
                    hidden_state_size   = args.hidden_state.copy(),
                    encoded_state_size  = args.encoded_state.copy(),
                    num_grids           = args.num_grids,
                    grid_min            = args.grid_min,
                    grid_max            = args.grid_max,
                    inv_denominator     = args.inv_denominator,
                    mode                = args.mode,
                    residual            = args.residual,
                    dynamic             = args.dynamic,
                    dropout_rate        = args.dropout,
                ),
                object_to_config(
                    Parameterizer,
                    type_to_config(RangeTransform),
                    data_min          = [False, 0.],
                    data_max          = [False, 1.],
                    target_min        = [True , 0.],
                    target_max        = [True , 1.],
                ),
                # torch.nn.Sigmoid,
                torch.nn.Tanh,
                object_to_config(
                    RangeTransform,
                    data_min          = -1,
                    data_max          = 1,
                    target_min        = 0,
                    target_max        = 1,
                ),
                target_name           ='model',
            ),
    ))
    tmp = check_config(model_config, locals=get_locals(custom_model))
    img_enc = instantiate(tmp,'img_enc')
    img_dec = instantiate(tmp,'img_dec')
    model   = build_train_1_model(
        img_enc = img_enc,
        img_dec = img_dec,
    )
    pdir, model_config['hash'] = build_model_dir(
        model,
        top_dir         = args.dest_top_dir,
        test_version    = args.test_version,
        training_stage  = 1,
    )

    if args.hash:
        print(model_config['hash'])

    if args.export :
        os.makedirs(os.path.dirname(pdir), exist_ok=True)
        pdir = save_config(model_config, pdir)
        
    if not args.hash:
        print(pdir)
