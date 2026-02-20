#! /usr/bin/env python3   
from typing import Literal
import sys, os
import tables

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import pandas as pd
import numpy as np
import json

__dataset_dir = os.path.join(THIS_DIR,'dataset')

def create_labels(
    features,
    labels,
    label_enumeration : Literal['linear', 'exponential'] = 'linear',
    unify_labels = False,
    force = False,
    save  = True,
    ) -> dict:
    # print(df.dtypes)
    json_path = os.path.join(__dataset_dir,'labels.json')
    if not save or force or not os.path.exists(json_path):
        label_dict = {}
        
        # Assume only binary categories exist
        cols = [
            _ for _ in features
                if '_is' in _
        ]
        if unify_labels and len(labels) == 2:
            cols += ['Label']
        for col in cols:
            label_dict[col] = {}
            if unify_labels and col == 'Label':
                vals =  labels
            else :
                vals = [False, True]
            for idx, val in enumerate(vals, start=(label_enumeration == 'linear')):
                label_dict[col][val] = 2 ** idx if label_enumeration == 'exponential' else \
                                       idx      if len(vals) > 2 else \
                                       bool(idx-1)
        if save:
            with open(os.path.join(__dataset_dir,'labels.json'), 'w') as fw:
                json.dump(label_dict, fw, indent=4)
            
    if save:
        with open(os.path.join(__dataset_dir,'labels.json'), 'r') as fr:
            label_dict = json.load(fr)
        
    return label_dict

def get_features_labels(
    file_name, 
    features, 
    labels, 
    remove_mass_pt_window=False,
    start = None,
    end = None,
):
    # load file
    with tables.open_file(file_name, 'r') as h5file:
        if start is None and end is None:
            njets = getattr(h5file.root,features[0]).shape[0]
        elif start is not None and end is None :
            njets = getattr(h5file.root,features[0])[start:].shape[0]
        elif start is None and end is not None :
            njets = getattr(h5file.root,features[0])[:end].shape[0]
        else :
            njets = getattr(h5file.root,features[0])[start:end].shape[0]

        # allocate arrays
        feature_array = np.zeros((njets,len(features)))
        label_array = np.zeros((njets,len(labels)))

        # load feature arrays
        for (i, feat) in enumerate(features):
            if start is None and end is None:
                feature_array[:,i] = getattr(h5file.root,feat)[:]
            elif start is not None and end is None :
                feature_array[:,i] = getattr(h5file.root,feat)[start:]
            elif start is None and end is not None :
                feature_array[:,i] = getattr(h5file.root,feat)[:end]
            else :
                feature_array[:,i] = getattr(h5file.root,feat)[start:end]

        # load labels arrays
        for (i, label) in enumerate(labels):
            prods = label.split('*')
            prod0 = prods[0]
            prod1 = prods[1]
            if start is None and end is None:
                fact0 = getattr(h5file.root,prod0)[:]
                fact1 = getattr(h5file.root,prod1)[:]
            elif start is not None and end is None :
                fact0 = getattr(h5file.root,prod0)[start:]
                fact1 = getattr(h5file.root,prod1)[start:]
            elif start is None and end is not None :
                fact0 = getattr(h5file.root,prod0)[:end]
                fact1 = getattr(h5file.root,prod1)[:end]
            else :
                fact0 = getattr(h5file.root,prod0)[start:end]
                fact1 = getattr(h5file.root,prod1)[start:end]
            label_array[:,i] = np.multiply(fact0,fact1)

        cond = np.sum(label_array,axis=1)==1
        # remove samples outside mass/pT window
        if remove_mass_pt_window:
            spec_array = np.zeros((njets,len(remove_mass_pt_window)))
            # load spectator arrays
            for (i, spec) in enumerate(remove_mass_pt_window.keys()):
                if start is None and end is None:
                    spec_array[:,i] = getattr(h5file.root,spec)[:]
                elif start is not None and end is None :
                    spec_array[:,i] = getattr(h5file.root,spec)[start:]
                elif start is None and end is not None :
                    spec_array[:,i] = getattr(h5file.root,spec)[:end]
                else :
                    spec_array[:,i] = getattr(h5file.root,spec)[start:end]
                    
            cond = np.stack([cond] + [
                cond_i 
                    for (low_lim, high_lim), spec_i in zip(
                        remove_mass_pt_window.values(), 
                        spec
                    )
                    for cond_i in [
                        low_lim < spec_i,
                        spec_i  < high_lim 
                    ]  
            ], axis = 0
            ).all(axis = 0)
            
        feature_array = feature_array[cond]
        label_array = label_array[cond]

    return feature_array, label_array

def build_dataset(force = False) -> None:
    dataset_path = os.path.join(__dataset_dir,'HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_h5_file_index.txt')
    if force or not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        ulr_link ='https://opendata.cern.ch/record/12102/file_index/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_h5_file_index.txt'
        os.system(' '.join([
            'wget',
            ulr_link,
            '-O' ,
            dataset_path
        ]))
        
        with open(dataset_path, 'r') as fr:
            train_links = [_.rstrip('\r\n') for _ in fr.readlines()]
    
        ulr_link = ulr_link.replace('train','test')
        os.system(' '.join([
            'wget',
            ulr_link,
            '-O' ,
            dataset_path.replace('train','test')
        ]))
        with open(dataset_path.replace('train','test'), 'r') as fr:
            test_links = [_.rstrip('\r\n') for _ in fr.readlines()]
           
        os.makedirs(os.path.join(__dataset_dir,"train"), exist_ok=True) 
        os.makedirs(os.path.join(__dataset_dir,"test"), exist_ok=True) 
        
        from multiprocessing import Pool
        
        download = lambda link, dataset : ' '.join([
            'wget',
            link,
            # '--progress=off',
            '-O' ,
            os.path.join(__dataset_dir,dataset,os.path.basename(link))
        ])
        with Pool(processes=2) as pool:
            pool.map(os.system, [download(*_) for _ in [
                (link, "train") for link in train_links
            ] + [
                (link, "test") for link in test_links
            ]],
                chunksize=1
            )
    return

def get_dataset_paths(
    subset : Literal['train','val','test'] | list[Literal['train','val','test']], 
    split = [0.5,0.5]
) :
    test = None
    train = None
    val = None
    
    if subset == 'test' or (isinstance(subset, list) and 'test' in subset):
        test_dir = os.path.join(__dataset_dir, 'test')
        test = [
            os.path.join(test_dir, _) for _ in os.listdir(test_dir)
        ]
    if subset == 'test':
        return test
    
    train_dir = os.path.join(__dataset_dir, 'train')
    files = [
        os.path.join(train_dir, _) for _ in os.listdir(train_dir)
    ]
    np.random.shuffle(files)
    split  = np.array(split[:2])
    split /= split.sum()
    split  = (split[0] * len(files)).round().astype(int)
    train, val = files[:split], files[split:]
    
    if subset == 'train':
        return train
    elif subset == 'val':
        return val
    
    out_dict = {'test' : test, 'train' : train, 'val' : val}
    return [out_dict[_] for _ in subset]

def get_dataset(subset, features, labels, remove_mass_pt_window, split = None):
    dataset_paths = get_dataset_paths(subset, **(dict(split=split) if split is not None else {}))
    if isinstance(dataset_paths[0], list):
        dataset_paths = [_ for _subset in dataset_paths for _ in _subset]
        
    df = []
    for dataset_path in dataset_path:
        feature_array, label_array = get_features_labels(
            file_name               = dataset_path, 
            features                = features, 
            labels                  = labels, 
            remove_mass_pt_window   = remove_mass_pt_window,
        )
        df.append( 
            pd.concat([
                pd.DataFrame(columns = features, data=feature_array),
                pd.DataFrame(columns = labels, data=label_array),
            ], join='outer', axis=1)
        )
    df = pd.concat(df, axis=0, ignore_index=True)
    return df

def get_columns():
    sample_file = os.path.join(
        __dataset_dir,
        'train',
        os.listdir(os.path.join(__dataset_dir,'train'))[0]
    )
    with tables.open_file(sample_file, 'r') as h5file:
        columns = getattr(h5file.root, '__members__')
    return columns

if __name__ == '__main__':
    # Download latest version
    df = build_dataset()
    print(get_columns())
    label_dict = create_labels(
        get_columns(), 
        labels = [],
        save   = True
    )
    for key, val in label_dict.items():
        print(key)
        for key, _val in val.items():
            print('  ', _val, key)
            
    exit()
    