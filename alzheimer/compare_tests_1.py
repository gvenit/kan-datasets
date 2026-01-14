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
    test_root = os.path.join(args.test_dir,'train_1')
    for root, dirs, files in os.walk(test_root):
        x = os.path.relpath(root,test_root)
        if len(x.split(os.sep)) == 3 and os.path.isfile(os.path.join(root, 'history.json')):
            tests.append(x.split(os.sep))
            
    # print(tests)
    
    tests = pd.DataFrame(data=tests, columns=['Model','Training','Version'])
    
    for idx in tests.index:
        x = os.path.join(
            tests.loc[idx, 'Model'],
            tests.loc[idx, 'Training'],
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
            tests.loc[idx,[metric]] = history[metric]
    
    print(tests)
    tests['Training'] = [_[:args.limit] for _ in tests['Training'].values]
    tests['Model']    = [_[:args.limit] for _ in tests['Model'].values]
    
    plt_dir = os.path.join(args.test_dir, 'comparison','train_1')
    os.makedirs(plt_dir, exist_ok=True)
    
    for col in tests.columns[3:]:
        # print(col, pd.isna(tests[col]).sum(), len(tests) - pd.isna(tests[col]).sum())
        if len(tests) - pd.isna(tests[col]).sum() < 2:
            tests.drop(columns=col, inplace=True)
    
    # Global Compare 
    # for col in tests.columns[3:]:
    ax = tests.set_index(list(tests.columns[:3])).plot.barh(
        # y        = col,
        y        = tests.columns[3:],
        subplots = True,
        legend   = False,
        sharex   = False,
        figsize  = (3 + len(tests.columns[3:]), 1.5+0.75*len(tests)),
        # figsize  = (1+0.5*len(tests),3 + len(tests.columns[3:])),
    )
    plt.xticks(rotation = 45, fontsize=7) 
    [axi.minorticks_on() for axi in ax]
    [axi.xaxis.grid(True, which='major') for axi in ax]
    [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir,f'global.png'))
    # plt.savefig(os.path.join(plt_dir,f'global_{col}.png'))
    plt.close('all')
    
    # Model Config Compare 
    for key, grp in tests.set_index(list(tests.columns[1:3])).groupby(tests.columns[0]):
        # print(grp)
        ax = grp.sort_index().plot.barh(
            y        = tests.columns[3:],
            subplots = True,
            legend   = False,
            sharex   = False,
            title    = key,
            figsize  = (3 + len(tests.columns[3:]), 1.5+0.75*len(grp)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'models_{key}.png'))
        plt.close('all')
        
    # Training Config Compare 
    for key, grp in tests.set_index([tests.columns[0],tests.columns[2]]).groupby(tests.columns[1]):
        # print(grp)
        ax = grp.sort_index().plot.barh(
            y        = tests.columns[3:],
            subplots = True,
            legend   = False,
            sharex   = False,
            title    = key,
            figsize  = (3 + len(tests.columns[3:]), 1.5+0.75*len(grp)),
        )
        plt.xticks(rotation = 45, fontsize=7) 
        [axi.minorticks_on() for axi in ax]
        [axi.xaxis.grid(True, which='major') for axi in ax]
        [axi.xaxis.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5) for axi in ax]
        
        plt.tight_layout()
        plt.savefig(os.path.join(plt_dir,f'trains_{key}.png'))
        plt.close('all')