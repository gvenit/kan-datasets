#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
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
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--actf', dest='actf', type=str, default='ReLU')
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
    # from custom_callbacks import *
    
    from build_model import build_train_1_model, build_model_dir

    args.test_version = '_'.join(['test',args.test_version])

    model_config = {}
    model_config['input_img_dim']  = [args.input_depth, args.input_height, args.input_width]
    model_config['output_img_dim'] = [args.output_depth, args.output_height, args.output_width]
    model_config['encoded_state']  = args.hidden_layers[-1]
    model_config['input']  = ['Path']
    model_config['output'] = ['Path']

    args.hidden_layers   = [args.input_width * args.input_height, *args.hidden_layers]
    # args.num_grids       = expand_value(args.num_grids,       len(args.hidden_layers)-1)
    # args.grid_min        = expand_value(args.grid_min,        len(args.hidden_layers)-1)
    # args.grid_max        = expand_value(args.grid_max,        len(args.hidden_layers)-1)
    # args.inv_denominator = expand_value(args.inv_denominator, len(args.hidden_layers)-1)

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
                    RangeTransform,
                    data_min          = 0,
                    data_max          = 1,
                    target_min        = -1,
                    target_max        = 1,
                ), *[
                    object_to_config(
                        torch.nn.Sequential,
                        object_to_config(
                            torch.nn.Linear,
                            in_features,
                            out_features,
                        ),
                        find_object_from_name(args.actf),
                        object_to_config(torch.nn.Dropout, args.dropout),
                    ) for in_features, out_features in zip(
                        args.hidden_layers[:-2], args.hidden_layers[1:-1],
                    )
                ], object_to_config(
                    torch.nn.Linear,
                    args.hidden_layers[-2], 
                    args.hidden_layers[-1],
                ),
                target_name     ='model',
            ),
    ))

    # Reverse order for decoder
    args.hidden_layers.reverse()   
    args.hidden_layers = args.hidden_layers[:-1] + [args.output_width * args.output_height]

    model_config.update(
        object_to_config(
            SubBatch,
            target_name         ='img_dec',
            input_data_dim      = -1,
            output_data_dim     = -2,
            **object_to_config(
                torch.nn.Sequential,*[
                    object_to_config(
                        torch.nn.Sequential,
                        object_to_config(
                            torch.nn.Linear,
                            in_features,
                            out_features,
                        ),
                        find_object_from_name(args.actf),
                        object_to_config(torch.nn.Dropout, args.dropout),
                    ) for in_features, out_features in zip(
                        args.hidden_layers[:-2], args.hidden_layers[1:-1],
                    )
                ], object_to_config(
                    torch.nn.Linear,
                    args.hidden_layers[-2], 
                    args.hidden_layers[-1],
                ),
                object_to_config(
                    Parameterizer,
                    type_to_config(RangeTransform),
                    data_min          = [False, 0.],
                    data_max          = [False, 1.],
                    target_min        = [True , 0.],
                    target_max        = [True , 1.],
                ),
                object_to_config(
                    LambdaModule,
                    'lambda x : 1. - torch.nn.functional.tanh(x)**2'
                ), 
                # torch.nn.Sigmoid,
                torch.nn.Tanh,
                # object_to_config(
                #     RangeTransform,
                #     data_min          = -1,
                #     data_max          = 1,
                #     target_min        = 0,
                #     target_max        = 1,
                # ),
                object_to_config(
                    Reshaper,
                    input_data_shape  = args.hidden_layers[-1:],
                    output_data_shape = model_config['output_img_dim'][1:],
                ),
                target_name     ='model',
            ),
    ))
    tmp = check_config(model_config)
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
