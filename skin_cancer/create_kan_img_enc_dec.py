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
    parser.add_argument('--input-height', '--height', dest='height', type=int, default=450)
    parser.add_argument('--slice-height', '--slice_height', dest='slice_height', type=int, default=16)
    parser.add_argument('--input-width', '--width', dest='width', type=int, default=600)
    parser.add_argument('--slice-width', '--slice_width', dest='slice_width', type=int, default=16)
    parser.add_argument('--input-channels', '--channels', dest='channels', type=int, default=3)
    parser.add_argument('--output-channels','--output_channels', dest='output_channels', type=int, default=1)
    parser.add_argument('--attention-size','--attention_size','--attention', dest='attention', type=int, default=4)
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='inv_denominator', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='RSWAFF')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()
    args.hidden_layers = [] if args.hidden_layers is None else args.hidden_layers

    import torch
    from collections import OrderedDict

    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.models import *
    from kan_utils.utils import expand_value
    # from custom_callbacks import *
    
    from build_model import build_model, build_model_dir, check_heads
    from prepare_dataset import get_dataset, expand_df_labels
    from custom_model import ImageSplitter, ImageMerger, AbstractSelfAttention
    import custom_model

    args.test_version = '_'.join(['test',args.test_version])
    
    df  = expand_df_labels(get_dataset())

    model_config = {}
    # model_config['input_img_dim']  = [args.channels, 300, 300]
    model_config['input_img_dim']  = [args.channels, args.height, args.width]
    model_config['sliced_img_dim'] = [args.channels, args.slice_height, args.slice_width]
    model_config['output_img_dim'] = {}
    model_config['encoded_state']  = args.hidden_layers[-1]
    model_config['path_cols']      = {'image' :['Image'], 'mask' : ['Mask']}
    model_config['input']  = ['Image']
    model_config['heads']  = []
    model_config['output'] = {}
    
    args.hidden_layers   = [args.slice_width * args.slice_height, *args.hidden_layers]
    args.num_grids       = expand_value(args.num_grids,       len(args.hidden_layers)-1)
    args.grid_min        = expand_value(args.grid_min,        len(args.hidden_layers)-1)
    args.grid_max        = expand_value(args.grid_max,        len(args.hidden_layers)-1)
    args.inv_denominator = expand_value(args.inv_denominator, len(args.hidden_layers)-1)

    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            object_to_config(
                OrderedDict,
                **object_to_config(
                    ImageSplitter,
                    target_name     ='split',
                    input_shape     = model_config['input_img_dim'][-2:],
                    output_shape    = model_config['sliced_img_dim'][-2:],
                    stride_percent  = 0.5,
                    output_dim      = 1,
                    keep_chn_dim    = True,
                ),
                **object_to_config(
                    Reshaper,
                    input_data_shape  = model_config['sliced_img_dim'][-2:],
                    output_data_shape = args.hidden_layers[:1],
                    target_name ='flatten',
                ),
                **object_to_config(
                    AbstractSelfAttention,
                    target_name     = 'attention',
                    key_model       = object_to_config(
                        FasterKAN,
                        hidden_layers   = [*args.hidden_layers[:-1],args.attention],
                        num_grids       = args.num_grids.copy(),
                        grid_min        = args.grid_min.copy(),
                        grid_max        = args.grid_max.copy(),
                        inv_denominator = args.inv_denominator.copy(),
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    query_model     = object_to_config(
                        FasterKAN,
                        hidden_layers   = [*args.hidden_layers[:-1],args.attention],
                        num_grids       = args.num_grids.copy(),
                        grid_min        = args.grid_min.copy(),
                        grid_max        = args.grid_max.copy(),
                        inv_denominator = args.inv_denominator.copy(),
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    value_model     = object_to_config(
                        FasterKAN,
                        hidden_layers   = args.hidden_layers.copy(),
                        num_grids       = args.num_grids.copy(),
                        grid_min        = args.grid_min.copy(),
                        grid_max        = args.grid_max.copy(),
                        inv_denominator = args.inv_denominator.copy(),
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    sequence_dim    = [1, -2],
                    input_data_dim  = -1,
                    output_data_dim = -1,
                    dropout_rate    = args.dropout,
                    reduction       = 'none',
                ),
                **object_to_config(
                    torch.nn.Dropout2d,
                    args.dropout,
                    target_name     ='dropout',
                ),
            ),
            target_name         ='Image Encoder',
    ))

    # Reverse order for decoder
    args.hidden_layers.reverse()   
    args.num_grids.reverse()       
    args.grid_min.reverse()        
    args.grid_max.reverse()        
    args.inv_denominator.reverse() 
    args.hidden_layers = args.hidden_layers[:-1] + [args.slice_width * args.slice_height]

    # Segmentor
    model_config['heads'].append('Segmentor')
    model_config['output'].update({
        'Segmentor' : ['Mask',] if args.output_channels == 1 else ['Image',],
    })
    check_heads(model_config, args)
    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            object_to_config(
                OrderedDict,
                **object_to_config(
                    AbstractSelfAttention,
                    target_name     ='greyscale' if args.output_channels == 1 else "RGB",
                    key_model       = object_to_config(
                        FasterKAN,
                        hidden_layers   = [model_config['encoded_state'],args.attention],
                        num_grids       = args.num_grids[-1],
                        grid_min        = args.grid_min[-1],
                        grid_max        = args.grid_max[-1],
                        inv_denominator = args.inv_denominator[-1],
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    query_model     = object_to_config(
                        FasterKAN,
                        hidden_layers   = [model_config['encoded_state'],args.attention],
                        num_grids       = args.num_grids[-1],
                        grid_min        = args.grid_min[-1],
                        grid_max        = args.grid_max[-1],
                        inv_denominator = args.inv_denominator[-1],
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    value_model     = object_to_config(
                        FasterKAN,
                        hidden_layers   = args.hidden_layers.copy(),
                        num_grids       = args.num_grids.copy(),
                        grid_min        = args.grid_min.copy(),
                        grid_max        = args.grid_max.copy(),
                        inv_denominator = args.inv_denominator.copy(),
                        mode            = args.mode,
                        residual        = args.residual,
                        dynamic         = args.dynamic,
                        dropout_rate    = args.dropout,
                    ),
                    sequence_dim    = -2,
                    input_data_dim  = -1,
                    output_data_dim = -1,
                    dropout_rate    = args.dropout,
                    reduction       = 'mean' if args.output_channels == 1 else 'none',
                    keepdim         = len(model_config['output_img_dim']['Segmentor']) > 2,
                ),
                **object_to_config(
                    Reshaper,
                    target_name       ='reconstruct',
                    input_data_shape  = args.hidden_layers[-1:],
                    output_data_shape = model_config['sliced_img_dim'][-2:],
                ),
                **object_to_config(
                    ImageMerger,
                    target_name     ='merge',
                    input_shape     = model_config['sliced_img_dim'][-2:],
                    output_shape    = model_config['output_img_dim']['Segmentor'][-2:],
                    stride_percent  = 0.5,
                    input_dim       = 1,
                    keep_chn_dim    = True,
                ),
                RSWAFF = RSWAFF,
                # **object_to_config(
                #     Parameterizer,
                #     type_to_config(RangeTransform),
                #     data_min          = [False, 0.],
                #     data_max          = [False, 1.],
                #     target_min        = [True , 0.],
                #     target_max        = [True , 1.],
                #     target_name       ='scale',
                # ),
                # **object_to_config(
                #     torch.nn.Tanh,
                #     target_name       ='normalize',
                # ),
                # **object_to_config(
                #     RangeTransform,
                #     data_min          = -1,
                #     data_max          = 1,
                #     target_min        = 0,
                #     target_max        = 1,
                #     target_name       ='pixelize',
                # ),
            ),
            target_name         ='Segmentor',
    ))
    
    # Classifier
    model_config['heads'].append('Classifier')
    model_config['output'].update({
        'Classifier' : df.drop(columns=['Image', 'Mask']).columns
    })
    check_heads(model_config, args)
    args.hidden_layers = args.hidden_layers[:-1] + [len(model_config['output']['Classifier'])]
    model_config.update(
        object_to_config(
            torch.nn.Sequential,
            # object_to_config(
            #     SubBatch,
            #     input_data_dim  = -1,
            #     model           = object_to_config(
            #         FasterKAN,
            #         hidden_layers   = args.hidden_layers.copy(),
            #         num_grids       = args.num_grids.copy(),
            #         grid_min        = args.grid_min.copy(),
            #         grid_max        = args.grid_max.copy(),
            #         inv_denominator = args.inv_denominator.copy(),
            #         mode            = args.mode,
            #         residual        = args.residual,
            #         dynamic         = args.dynamic,
            #         dropout_rate    = args.dropout,
            #     ),
            # object_to_config(
            #     LambdaModule,
            #     'lambda x: x.mean(dim=[1, -2])'
            # ),
            object_to_config(
                AbstractSelfAttention,
                key_model       = object_to_config(
                    FasterKAN,
                    hidden_layers   = [*args.hidden_layers[:-1],args.attention],
                    num_grids       = args.num_grids.copy(),
                    grid_min        = args.grid_min.copy(),
                    grid_max        = args.grid_max.copy(),
                    inv_denominator = args.inv_denominator.copy(),
                    mode            = args.mode,
                    residual        = args.residual,
                    dynamic         = args.dynamic,
                    dropout_rate    = args.dropout,
                ),
                query_model     = object_to_config(
                    FasterKAN,
                    hidden_layers   = [*args.hidden_layers[:-1],args.attention],
                    num_grids       = args.num_grids.copy(),
                    grid_min        = args.grid_min.copy(),
                    grid_max        = args.grid_max.copy(),
                    inv_denominator = args.inv_denominator.copy(),
                    mode            = args.mode,
                    residual        = args.residual,
                    dynamic         = args.dynamic,
                    dropout_rate    = args.dropout,
                ),
                value_model     = object_to_config(
                    FasterKAN,
                    hidden_layers   = args.hidden_layers.copy(),
                    num_grids       = args.num_grids.copy(),
                    grid_min        = args.grid_min.copy(),
                    grid_max        = args.grid_max.copy(),
                    inv_denominator = args.inv_denominator.copy(),
                    mode            = args.mode,
                    residual        = args.residual,
                    dynamic         = args.dynamic,
                    dropout_rate    = args.dropout,
                ),
                sequence_dim    = [1, -2],
                input_data_dim  = -1,
                output_data_dim = -1,
                dropout_rate    = args.dropout,
                reduction       = 'mean',
            ),
            RSWAFF,
            target_name     ='Classifier',
    ))
    
    # Validate & Export
    tmp = check_config(model_config, get_locals(custom_model))
    img_enc = instantiate(tmp,'Image Encoder')
    heads   = {
        head : instantiate(tmp, head)
            for head in model_config['heads']
    }
    model   = build_model(
        img_enc = img_enc,
        **heads,
    )
    pdir, model_config['hash'] = build_model_dir(
        model,
        top_dir         = args.dest_top_dir,
        test_version    = args.test_version,
    )
    if args.hash:
        print(model_config['hash'])

    if args.export :
        os.makedirs(os.path.dirname(pdir), exist_ok=True)
        pdir = save_config(model_config, pdir)
        
    if not args.hash:
        print(pdir)
