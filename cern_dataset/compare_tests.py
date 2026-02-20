#!/usr/bin/env python3   
import sys, os
from argparse import ArgumentParser

THIS_DIR = os.path.dirname(__file__)
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
    
    for idx in tests.index:
        x = os.path.join(
            tests.loc[idx, 'Configuration'],
            tests.loc[idx, 'Version'],
        )
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
    tests.dropna(axis=1, how='all', inplace=True)
    print(tests)
    
    plt_dir = os.path.join(args.test_dir, 'comparison')
    os.makedirs(plt_dir, exist_ok=True)
    
    for col in tests.columns[2:]:
        # print(col, pd.isna(tests[col]).sum(), len(tests) - pd.isna(tests[col]).sum())
        if len(tests) - pd.isna(tests[col]).sum() < 2:
            tests.drop(columns=col, inplace=True)
    
    # Global Compare 
    tests_g = tests.set_index(list(tests.columns[:2]))
    ax = tests_g.plot.barh(
        # y        = col,
        y        = tests.columns[2:],
        subplots = True,
        legend   = False,
        sharex   = False,
        figsize  = (3 + len(tests.columns[2:]), 1.5+0.75*len(tests)),
    )
    plt.xticks(rotation = 45, fontsize=7) 
    [axi.set(xlim=(0.999*tests_g[col].min(), 1.001*tests_g[col].max())) for axi, col in zip(ax, tests.columns[2:])]
    [axi.minorticks_on() for axi in ax]
    [axi.xaxis.grid(True, which='major') for axi in ax]
    [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir,f'global.png'))
    # plt.savefig(os.path.join(plt_dir,f'global_{col}.png'))
    plt.close('all')
    
    for col in tests.columns[2:]:
        print('-- Metric :', col)
        print(f'  -- Max : {tests_g[col].max() :.4f} --  {tests_g.index[tests_g[col].argmax()]}')
        print(f'  -- Min : {tests_g[col].min() :.4f} --  {tests_g.index[tests_g[col].argmin()]}')
    