#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__' :
    parser = ArgumentParser(
        description='Spatial Encoder/Decoder model configuration script for the Alzheimer\'s Dataset.'
    )

    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--img','--image-config', dest='img_hash', help='The hash of the image encoder/decoder model configuration file.')
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()

    # import torch

    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.models import *
    from kan_utils.utils import expand_value
    # from custom_callbacks import *
    
    import custom_model
    from build_model import get_model_config_path, build_train_2_model, build_model_dir

    args.test_version = '_'.join(['test',args.test_version])

    model_config = {}
    model_config['img_hash'] = args.img_hash
    img_config   = load_config(
        get_model_config_path(
            training_stage  = 1,
            model_hash      = model_config['img_hash'],
            top_dir         = args.dest_top_dir,
            test_version    = args.test_version,
        ),
        locals=get_locals(custom_model)
    )
    model_config['encoded_state']  = args.hidden_layers[-1]
    model_config['input']  = ['Path']
    model_config['output'] = ['Path']
    
    hidden_size = max(args.hidden_layers[0],model_config['encoded_state'])
    proj_size   = model_config['encoded_state'] if model_config['encoded_state'] < hidden_size else 0

    model_config.update(
        object_to_config(
            SubBatch,
            target_name     ='spt_enc',             
            input_data_dim  = -2,
            output_data_dim = -1,
            **object_to_config(
                torch.nn.Sequential,
                object_to_config(
                    custom_model.LSTMEncoder,
                    input_size      = img_config['encoded_state'],
                    hidden_size     = hidden_size,
                    num_layers      = len(args.hidden_layers),
                    batch_first     = True,
                    dropout         = args.dropout,
                    bidirectional   = False,
                    proj_size       = proj_size,
                    return_sequence = False,
                    return_states   = False,
                    trainable_states= True,
                ),
                torch.nn.Tanh,
                target_name     = 'model',
            ),
    ))
    
    hidden_size = max(int(args.hidden_layers[0]),int(img_config['encoded_state']))
    proj_size   = int(img_config['encoded_state']) if int(img_config['encoded_state']) < hidden_size else 0
    
    if model_config['encoded_state'] == proj_size:
        feedback = object_to_config(
            torch.nn.Tanh,
            target_name = 'feedback',
        )
    else :
        feedback = object_to_config(
            torch.nn.Sequential,
            object_to_config(
                torch.nn.Linear,
                int(img_config['encoded_state']),
                model_config['encoded_state'],
            ),
            torch.nn.Tanh,
            target_name = 'feedback',
        )
        
    model_config.update(
        object_to_config(
            SubBatch,
            target_name     ='spt_dec',
            input_data_dim  = -1,
            output_data_dim = -2,
            **object_to_config(
                torch.nn.Sequential,
                object_to_config(
                    custom_model.LSTMDecoder,
                    **feedback,
                    input_size      = model_config['encoded_state'],
                    hidden_size     = hidden_size,
                    num_layers      = len(args.hidden_layers),
                    batch_first     = True,
                    dropout         = args.dropout,
                    proj_size       = proj_size,
                    return_sequence = True,
                    return_states   = False,
                    trainable_states= True,
                    seq_len         = img_config['output_img_dim'][0],
                ),
                torch.nn.Tanh,
                target_name     = 'model',
            ),
    ))
    tmp = check_config(model_config, locals=get_locals(custom_model))
    spt_enc = instantiate(tmp,'spt_enc')
    spt_dec = instantiate(tmp,'spt_dec')
    model   = build_train_2_model(
        spt_enc = spt_enc,
        spt_dec = spt_dec,
    )
    pdir, model_config['hash'] = build_model_dir(
        model,
        top_dir         = args.dest_top_dir,
        test_version    = args.test_version,
        training_stage  = 2,
    )
    
    if args.hash:
        print(model_config['hash'])

    if args.export :
        os.makedirs(os.path.dirname(pdir), exist_ok=True)
        pdir = save_config(model_config, pdir)
        
    if not args.hash:
        print(pdir)
