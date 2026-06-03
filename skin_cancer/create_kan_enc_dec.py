#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the Ship Performance Clusterring Dataset.'
    )
    parser.add_argument('-d', '--dest-top-directory', dest='dest_top_dir', default=os.path.join(THIS_DIR,'train'))
    parser.add_argument('--input-channels', '--channels', dest='channels', type=int, default=3)
    parser.add_argument('--output-channels','--output_channels', dest='output_channels', type=int, default=1)
    parser.add_argument('--resize', dest='resize', type=int, nargs=2, metavar=('H', 'W'), help="Resize images to HxW (e.g., --resize 16 16)")
    parser.add_argument('--layers', '--hidden-layers', dest='hidden_layers', action='extend', nargs="+")
    parser.add_argument('--num-grids', dest='num_grids', action='extend', nargs="+")
    parser.add_argument('--grid-min', dest='grid_min', action='extend', nargs="+")
    parser.add_argument('--grid-max', dest='grid_max', action='extend', nargs="+")
    parser.add_argument('--scale','--inv_denominator', dest='scale', action='extend', nargs="+")
    parser.add_argument('--mode', dest='mode', type=str, default='custom')
    parser.add_argument('--residual', dest='residual', action='store_true')
    parser.add_argument('--dynamic', dest='dynamic', action='store_true')
    parser.add_argument('--use-v2', dest='use_v2', action='store_true')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--dropout-linear', dest='dropout_linear', type=float, default=None)
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.add_argument('--no-normalize-rbf', dest='normalize_rbf', action='store_false', default=True)
    parser.add_argument('--hash', action='store_true',dest='hash', help="Return the corresponding hash value instead of the full directory")
    parser.add_argument('--export', action='store_true',dest='export', help="Save the model configuration")
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()
    args.hidden_layers = [] if args.hidden_layers is None else args.hidden_layers

    import numpy as np
    import pandas as pd
    
    from kan_utils.config import *
    from kan_utils.metrics import *
    from kan_utils.callbacks import UpdatableFloat
    from kan_utils.models import *
    from kan_utils.utils import expand_value
    
    from build_model import build_model, build_model_dir, check_heads
    from prepare_dataset import build_dataset, expand_df_labels
    
    args.test_version = '_'.join(['test',args.test_version])
    
    df : pd.DataFrame = expand_df_labels(build_dataset())
    labels = list(filter(lambda x: 'Label_Is_' in x, df.columns.tolist()))

    if args.resize is not None:
        h, w = args.resize
        num_features = h * w
    else:
        num_features = 600*450 
        
    layer_split = len(args.hidden_layers) // 2 + 1

    model_config = {}
    model_config['input_img_dim']  = [args.channels, *args.resize]
    model_config['encoded_state']  = len(labels) if len(args.hidden_layers) == 0 else \
                                     args.hidden_layers[layer_split-1]
    model_config['path_cols']   = {'image' :['Image'], 'mask' : ['Mask']}
    model_config['input']       = ['Image']
    model_config['heads']       = []
    model_config['output']      = {}
    model_config['flatten']     = True
    
    args.hidden_layers      = [int(np.prod(model_config['input_img_dim'])), *args.hidden_layers, len(labels)]
    args.num_grids          = expand_value(args.num_grids,       len(args.hidden_layers)-1)
    args.grid_min           = expand_value(args.grid_min,        len(args.hidden_layers)-1)
    args.grid_max           = expand_value(args.grid_max,        len(args.hidden_layers)-1)
    args.scale              = expand_value(args.scale, len(args.hidden_layers)-1)

    # Encoder
    model_config.update(
        object_to_config(
            FasterKAN,
            hidden_layers   = args.hidden_layers[:layer_split+1],
            num_grids       = args.num_grids[:layer_split],
            grid_min        = args.grid_min[:layer_split],
            grid_max        = args.grid_max[:layer_split],
            inv_denominator = args.scale[:layer_split],
            mode            = args.mode,
            residual        = args.residual,
            dynamic         = args.dynamic,
            use_v2          = args.use_v2,
            normalize       = args.normalize,
            normalize_rbf   = args.normalize_rbf,
            dropout_rate    = object_to_config(
                                UpdatableFloat,
                                args.dropout,
                            ),
            dropout_linear  = None if args.dropout_linear is None else object_to_config(
                                UpdatableFloat,
                                args.dropout_linear,
                            ),
            target_name     ='Image Encoder',
    ))
    model_config['outputs_logits'] = True
    model_config['flatten'] = True
    
    # Classifier
    model_config['heads'].append('Classifier')
    model_config['output'].update({
        'Classifier' : df.drop(columns=['Image', 'Mask']).columns
    })
    check_heads(model_config, args)
    args.hidden_layers = args.hidden_layers[:-1] + [len(model_config['output']['Classifier'])]
    model_config.update(
        object_to_config(
            FasterKAN,
            hidden_layers   = args.hidden_layers[layer_split:],
            num_grids       = args.num_grids[layer_split:],
            grid_min        = args.grid_min[layer_split:],
            grid_max        = args.grid_max[layer_split:],
            inv_denominator = args.scale[layer_split:],
            mode            = args.mode,
            residual        = args.residual,
            dynamic         = args.dynamic,
            use_v2          = args.use_v2,
            normalize       = args.normalize,
            normalize_rbf   = args.normalize_rbf,
            dropout_rate    = object_to_config(
                                UpdatableFloat,
                                args.dropout,
                            ),
            dropout_linear  = None if args.dropout_linear is None else object_to_config(
                                UpdatableFloat,
                                args.dropout_linear,
                            ),
            target_name     ='Classifier',
    ))
    
    # Segmentor
    model_config['heads'].append('Segmentor')
    model_config['output'].update({
        'Segmentor' : ['Mask',] if args.output_channels == 1 else ['Image',],
    })
    check_heads(model_config, args)
    
    args.hidden_layers   = [int(np.prod(model_config['output_img_dim']['Segmentor'])), *args.hidden_layers[1:layer_split+1]]
    args.num_grids       = args.num_grids[:layer_split]
    args.grid_min        = args.grid_min[:layer_split]
    args.grid_max        = args.grid_max[:layer_split]
    args.scale = args.scale[:layer_split]
    
    args.hidden_layers.reverse()   
    args.num_grids.reverse()       
    args.grid_min.reverse()        
    args.grid_max.reverse()        
    args.scale.reverse() 

    model_config.update(
        object_to_config(
            FasterKAN,
            hidden_layers   = args.hidden_layers.copy(),
            num_grids       = args.num_grids.copy(),
            grid_min        = args.grid_min.copy(),
            grid_max        = args.grid_max.copy(),
            inv_denominator = args.scale.copy(),
            mode            = args.mode,
            residual        = args.residual,
            dynamic         = args.dynamic,
            use_v2          = args.use_v2,
            normalize       = args.normalize,
            normalize_rbf   = args.normalize_rbf,
            dropout_rate    = object_to_config(
                                UpdatableFloat,
                                args.dropout,
                            ),
            dropout_linear  = None if args.dropout_linear is None else object_to_config(
                                UpdatableFloat,
                                args.dropout_linear,
                            ),
            target_name     ='Segmentor',
    ))
    
    # Validate & Export
    tmp = check_config(model_config, get_locals())
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
