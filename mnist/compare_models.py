#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

if __name__ == '__main__' :
    parser = ArgumentParser(
        description='Training script for the Ship Performance Clusterring Dataset.'
    )
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('-l', '--limit', dest='limit', type=int, default=-1, help='Limit the number of characters of the hashes shown in the figures.')

    args = parser.parse_args()

    # Check argument validity
    if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)
        
    else :
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    import pandas as pd
    import matplotlib.pyplot as plt
    
    from kan_utils.utils import load_dict

    tests = []
    test_root = args.test_dir
    for root, dirs, files in os.walk(test_root):
        x = os.path.relpath(root,test_root)
        if len(x.split(os.sep)) == 2 and os.path.isfile(os.path.join(root, 'history.json')):
            tests.append(x.split(os.sep))
            
    # print(tests)
    
    tests = pd.DataFrame(data=tests, columns=['Configuration','Version']).sort_values(['Configuration','Version']).reset_index(drop=True)
    
    # Add columns for model configuration
    for col in ['num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'use_logits']:
        tests[col] = None
    
    for idx in tests.index:
        x = os.path.join(
            tests.loc[idx, 'Configuration'],
            tests.loc[idx, 'Version'],
        )
        
        # Load model configuration
        model_config_path = os.path.join(test_root, x, 'config', 'model.json')
        train_config_path = os.path.join(test_root, x, 'config', 'train.json')
        
        if os.path.exists(model_config_path):
            model_config = load_dict(model_config_path.replace('.json', ''))
            try:
                # Extract model parameters from the nested structure
                # model_args[0]['_args'] = [[['kan', {...}]]]
                model_args = model_config.get('model_args', [{}])[0]
                kan_config = None
                if '_args' in model_args and len(model_args['_args']) > 0:
                    # Go through the nested list structure
                    for outer_item in model_args['_args']:
                        if isinstance(outer_item, list):
                            for middle_item in outer_item:
                                if isinstance(middle_item, list) and len(middle_item) >= 2:
                                    if middle_item[0] == 'kan' and isinstance(middle_item[1], dict):
                                        kan_config = middle_item[1].get('_kwargs', {})
                                        break
                        if kan_config:
                            break
                
                if kan_config:
                    hidden_layers = kan_config.get('hidden_layers', [])
                    if hidden_layers:
                        tests.loc[idx, 'num_layers'] = len(hidden_layers) - 1
                        tests.loc[idx, 'hidden_layers'] = str(hidden_layers)
                    
                    mode = kan_config.get('mode', 'N/A')
                    tests.loc[idx, 'mode'] = mode
                    
                    grids = kan_config.get('num_grids', [])
                    tests.loc[idx, 'grids'] = str(grids) if grids else 'N/A'
                    
                    grid_min = kan_config.get('grid_min', [])
                    tests.loc[idx, 'grid_min'] = str(grid_min) if grid_min else 'N/A'
                    
                    grid_max = kan_config.get('grid_max', [])
                    tests.loc[idx, 'grid_max'] = str(grid_max) if grid_max else 'N/A'
                    
                    scale = kan_config.get('inv_denominator', [])
                    tests.loc[idx, 'scale'] = str(scale) if scale else 'N/A'
            except Exception as e:
                print(f'Warning: Could not parse model config for "{x}": {e}')
        
        if os.path.exists(train_config_path):
            train_config = load_dict(train_config_path.replace('.json', ''))
            # Check if BCEWithLogitsLoss is used (indicates logits)
            criterion_str = str(train_config)
            tests.loc[idx, 'use_logits'] = 'Yes' if 'BCEWithLogitsLoss' in criterion_str else 'No'
        
        history = load_dict(os.path.join(
            test_root,
            x,
            'history'
        ))
        if 'test' in history.keys():
            history = history['test']['best']
        else :
            print(f'Dropped "{x}"; no test was performed with this configuration.')
            tests.drop(idx, axis=0, inplace=True)
            continue
        for metric in history.keys():
            if metric not in tests.columns:
                tests[metric] = float('NaN')
            try :
                tests.loc[idx,[metric]] = history[metric]
            except:
                pass
    
    tests['Configuration'] = [_[:args.limit] for _ in tests['Configuration'].values]
    
    # Drop NaN columns but keep the configuration columns
    config_cols = ['num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'use_logits']
    other_cols = [col for col in tests.columns if col not in config_cols and col not in ['Configuration', 'Version']]
    for col in other_cols:
        if tests[col].isna().all():
            tests.drop(columns=col, inplace=True)
    
    # Print model configuration details
    print("\n" + "="*80)
    print("MODEL CONFIGURATIONS")
    print("="*80)
    for idx in tests.index:
        config_name = tests.loc[idx, 'Configuration']
        version = tests.loc[idx, 'Version']
        num_layers = tests.loc[idx, 'num_layers']
        hidden_layers = tests.loc[idx, 'hidden_layers']
        mode = tests.loc[idx, 'mode']
        grids = tests.loc[idx, 'grids']
        grid_min = tests.loc[idx, 'grid_min']
        grid_max = tests.loc[idx, 'grid_max']
        scale = tests.loc[idx, 'scale']
        use_logits = tests.loc[idx, 'use_logits']
        
        print(f"\n[{config_name} / {version}]")
        print(f"  Layers: {num_layers} | Hidden: {hidden_layers} | Mode: {mode} | Grids: {grids}")
        print(f"  GridMin: {grid_min} | GridMax: {grid_max} | Scale: {scale} | Logits: {use_logits}")
    
    print("\n" + "="*80)
    print(tests)
    print("="*80)
    
    plt_dir = os.path.join(args.test_dir, 'comparison')
    os.makedirs(plt_dir, exist_ok=True)
    
    # Get only metric columns (exclude configuration columns)
    config_cols = ['Configuration', 'Version', 'num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'use_logits']
    metric_cols = [col for col in tests.columns if col not in config_cols]
    
    for col in metric_cols:
        # print(col, pd.isna(tests[col]).sum(), len(tests) - pd.isna(tests[col]).sum())
        if len(tests) - pd.isna(tests[col]).sum() < 2:
            tests.drop(columns=col, inplace=True)
    
    # Update metric_cols after dropping
    metric_cols = [col for col in tests.columns if col not in config_cols]
    
    # Global Compare 
    tests_g = tests.set_index(list(tests.columns[:2]))
    if len(metric_cols) > 0:
        ax = tests_g.plot.barh(
            # y        = col,
            y        = metric_cols,
            subplots = True,
            legend   = False,
            sharex   = False,
            figsize  = (3 + len(metric_cols), 1.5+0.75*len(tests)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.set(xlim=(0.999*tests_g[col].min(), 1.001*tests_g[col].max())) for axi, col in zip(ax, metric_cols)]
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'global.png'))
        # plt.savefig(os.path.join(plt_dir,f'global_{col}.png'))
        plt.close('all')
        
    # Top 30 by F1Score
    if 'F1Score' in tests.columns and len(metric_cols) > 0:
        tests_top30 = tests.nlargest(30, 'F1Score')
        tests_top30_g = tests_top30.set_index(list(tests_top30.columns[:2]))
        
        ax = tests_top30_g.plot.barh(
            y        = metric_cols,
            subplots = True,
            legend   = False,
            sharex   = False,
            figsize  = (3 + len(metric_cols), 1.5+0.75*len(tests_top30)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.set(xlim=(0.999*tests_top30_g[col].min(), 1.001*tests_top30_g[col].max())) for axi, col in zip(ax, metric_cols)]
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'top30_f1score.png'))
        plt.close('all')
    
    # Print top 5 for key metrics
    print("\n" + "="*80)
    print("TOP 5 MODELS BY METRIC")
    print("="*80)
    
    key_metrics = ['Accuracy', 'F1Score', 'AUROC']
    for metric in key_metrics:
        if metric in tests.columns:
            print(f"\n{'='*80}")
            print(f"TOP 5 - {metric}")
            print('='*80)
            top5 = tests.nlargest(5, metric)[['Configuration', 'Version', metric, 'num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'use_logits']]
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                print(f"{i}. {row[metric]:.6f} - [{row['Configuration']} / {row['Version']}]")
                print(f"   Layers: {row['num_layers']} | Hidden: {row['hidden_layers']} | Mode: {row['mode']} | Grids: {row['grids']}")
                print(f"   GridMin: {row['grid_min']} | GridMax: {row['grid_max']} | Scale: {row['scale']} | Logits: {row['use_logits']}")
    
    print("\n" + "="*80)
    print("DETAILED METRIC SUMMARY")
    print("="*80)
    for col in metric_cols:
        print('-- Metric :', col)
        print(f'  -- Max : {tests_g[col].max() :.4f} --  {tests_g.index[tests_g[col].argmax()]}')
        print(f'  -- Min : {tests_g[col].min() :.4f} --  {tests_g.index[tests_g[col].argmin()]}')
    