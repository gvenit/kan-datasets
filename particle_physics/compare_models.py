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
    import matplotlib.ticker as ticker
    import warnings
    
    from kan_utils.utils import load_dict
    
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    ticker.Locator.MAXTICKS = 5000

    tests = []
    test_root = args.test_dir
    for root, dirs, files in os.walk(test_root):
        x = os.path.relpath(root,test_root)
        if len(x.split(os.sep)) == 2 and os.path.isfile(os.path.join(root, 'history.json')):
            tests.append(x.split(os.sep))
    
    tests = pd.DataFrame(data=tests, columns=['Configuration','Version']).sort_values(['Configuration','Version']).reset_index(drop=True)
    
    config_cols = ['num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'dropout', 'lr', 'weight_decay', 'use_logits']
    for col in config_cols:
        tests[col] = None
    
    skipped_models = 0
    
    for idx in tests.index:
        x = os.path.join(
            tests.loc[idx, 'Configuration'],
            tests.loc[idx, 'Version'],
        )
        
        # Check for hyperparameters.csv file
        hyperparams_csv_path = os.path.join(test_root, x, 'config', 'hyperparameters.csv')
        
        if not os.path.exists(hyperparams_csv_path):
            # print(f'Skipped "{x}"; hyperparameters.csv not found.')
            tests.drop(idx, axis=0, inplace=True)
            skipped_models += 1
            continue
        
        # Read hyperparameters from CSV
        try:
            hyperparams_df = pd.read_csv(hyperparams_csv_path)
            # Convert to dictionary for easier access
            hyperparams = dict(zip(hyperparams_df['parameter'], hyperparams_df['value']))
            
            # Extract model hyperparameters
            layers_str = hyperparams.get('layers', '')
            if layers_str:
                # Parse layers (e.g., "16 16" -> ['16', '16'])
                layers = str(layers_str).split()
                tests.loc[idx, 'num_layers'] = len(layers)
                tests.loc[idx, 'hidden_layers'] = str(layers)
            else:
                tests.loc[idx, 'num_layers'] = 'N/A'
                tests.loc[idx, 'hidden_layers'] = 'N/A'
            
            tests.loc[idx, 'mode'] = hyperparams.get('mode', 'N/A')
            tests.loc[idx, 'grids'] = hyperparams.get('num_grids', 'N/A')
            tests.loc[idx, 'grid_min'] = hyperparams.get('grid_min', 'N/A')
            tests.loc[idx, 'grid_max'] = hyperparams.get('grid_max', 'N/A')
            tests.loc[idx, 'scale'] = hyperparams.get('scale', 'N/A')
            tests.loc[idx, 'dropout'] = hyperparams.get('dropout', 'N/A')
            tests.loc[idx, 'lr'] = hyperparams.get('learning_rate', 'N/A')
            tests.loc[idx, 'weight_decay'] = hyperparams.get('weight_decay', 'N/A')
            
            # Check for logits
            with_logits = hyperparams.get('with_logits', '0')
            tests.loc[idx, 'use_logits'] = 'Yes' if str(with_logits) == '1' else 'No'
            
        except Exception as e:
            print(f'Warning: Could not parse hyperparameters.csv for "{x}": {e}')
            tests.drop(idx, axis=0, inplace=True)
            skipped_models += 1
            continue
        
        # Load history
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
    other_cols = [col for col in tests.columns if col not in config_cols and col not in ['Configuration', 'Version']]
    for col in other_cols:
        if tests[col].isna().all():
            tests.drop(columns=col, inplace=True)
    
    plt_dir = os.path.join(args.test_dir, 'comparison')
    os.makedirs(plt_dir, exist_ok=True)
    
    # Get only metric columns (exclude configuration columns)
    config_cols = ['Configuration', 'Version'] + config_cols
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
            top5 = tests.nlargest(5, metric)[['Configuration', 'Version', metric, 'num_layers', 'hidden_layers', 'mode', 'grids', 'grid_min', 'grid_max', 'scale', 'dropout', 'lr', 'weight_decay', 'use_logits']]
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                print(f"{i}. {row[metric]:.6f} - [{row['Configuration']} / {row['Version']}]")
                print(f"   Layers: {row['num_layers']} | Hidden: {row['hidden_layers']} | Mode: {row['mode']} | Grids: {row['grids']}")
                print(f"   GridMin: {row['grid_min']} | GridMax: {row['grid_max']} | Scale: {row['scale']}")
                print(f"   Dropout: {row['dropout']} | LR: {row['lr']} | WeightDecay: {row['weight_decay']} | Logits: {row['use_logits']}")
    
    print("\n" + "="*80)
    print("DETAILED METRIC SUMMARY")
    print("="*80)
    for col in metric_cols:
        print('-- Metric :', col)
        print(f'  -- Max : {tests_g[col].max() :.4f} --  {tests_g.index[tests_g[col].argmax()]}')
        print(f'  -- Min : {tests_g[col].min() :.4f} --  {tests_g.index[tests_g[col].argmin()]}')
    
    print(f"\nTotal models processed: {len(tests)}")
    print(f"Models skipped (missing hyperparameters.csv): {skipped_models}")

    