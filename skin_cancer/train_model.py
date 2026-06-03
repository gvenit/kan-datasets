#!/usr/bin/env python3   
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    TOP_DIR = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(
        description='Training script for the HAM10000 Skin Cancer Dataset.'
    )
    parser.add_argument('-d', '--test-dir', dest='test_dir', default=os.path.join(THIS_DIR,'train'), help='The directory to be used as a top directory for training.')
    parser.add_argument('-t', '--train-config', dest='train_config', help='The hash of the training configuration file.')
    parser.add_argument('-m', '--model-config', dest='model_config', help='The hash of the model configuration file.')
    parser.add_argument('--hash', dest='hash', type=str, help='The hash value of the unified configuration.')
    parser.add_argument('--test-version', dest='test_version', type=str, default='0')

    args = parser.parse_args()
    args.test_version = '_'.join(['test',args.test_version])

    # Check argument validity
    if  os.path.isdir(args.test_dir) or not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir, exist_ok=True)

    else :
        raise ValueError(f'Destination folder is not a directory; got "{os.path.splitext(args.test_dir)[-1]}"')

    # Handle unified --hash mode (from create_configs_logits.py)
    if args.hash is not None:
        args.test_dir = os.path.join(args.test_dir, args.hash, args.test_version)

        path = os.path.join(args.test_dir, 'config', 'train.json')
        if os.path.exists(path):
            args.train_config = path
            print(f'-- Using training configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate training configuration file in specified location; got {path}')

        path = os.path.join(args.test_dir, 'config', 'model.json')
        if os.path.exists(path):
            args.model_config = path
            print(f'-- Using model configuration path "{path}"')
        else :
            raise ValueError(f'Cannot locate model configuration file in specified location; got {path}')

    # Handle legacy -t/-m mode
    else:
        from build_model import get_model_config_path, get_train_config_path

        if args.model_config is None :
            raise ValueError(f'Cannot locate model configuration file.')

        else:
            args.model_config = get_model_config_path(
                training_stage  = 1,
                model_hash      = args.model_config,
                top_dir         = args.test_dir,
                test_version    = args.test_version,
            )
            args.model_config = f'{args.model_config}.json'

            if not os.path.exists(args.model_config) :
                raise ValueError(f'Cannot locate model configuration file in specified location; got {args.model_config}')
            else :
                print(f'-- Using model configuration path "{args.model_config}"')

        if args.train_config is None :
            raise ValueError(f'Cannot locate training configuration file.')

        else:
            args.train_config = get_train_config_path(
                training_stage  = 1,
                train_hash      = args.train_config,
                top_dir         = args.test_dir,
                test_version    = args.test_version,
            )
            args.train_config = f'{args.train_config}.json'

            if not os.path.exists(args.train_config) :
                raise ValueError(f'Cannot locate training configuration file in specified location; got {args.train_config}')
            else :
                print(f'-- Using training configuration path "{args.train_config}"')
             
    # import set_environment
    
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from kan_utils.config import *
    from kan_utils.dataset import smart_split_dataset
    from kan_utils.training import train
    from kan_utils.utils import set_seed, save_dict, load_dict
    from kan_utils.performance import get_summary

    from prepare_dataset import build_dataset, expand_df_labels, normalize_dataset, get_groups
    import extract_statistics
    # import custom_callbacks
    from build_model import build_model, get_training_subdir
    from custom_dataset import SkinCancerDataset, get_extra_transforms
    import custom_model
    

    device = torch.device(
        # 'cpu'
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Check configuration file validity
    train_config = load_config(args.train_config, locals=get_locals(extract_statistics))
    model_config = load_config(args.model_config, locals=get_locals(custom_model))
    set_seed(train_config['seed'])

    # Instantiate models
    img_enc = instantiate(model_config,'Image Encoder')
    heads   = {
        head : instantiate(model_config, head)
            for head in model_config['heads']
    }
    model   = build_model(
        img_enc = img_enc,
        **heads,
    )
    print('-- Model :', model)

    # Instantiate criterion
    criterion = instantiate(train_config,'criterion')
    print('-- Criterion :', criterion)

    # Instantiate optimizer
    optimizer = instantiate(train_config,'optimizer', model.parameters(), lr = train_config['lr'])
    print('-- Optimizer :', optimizer)

    # Instantiate scheduler
    scheduler = instantiate(train_config,'scheduler', optimizer)
    print('-- Scheduler :', scheduler)

    # Instantiate evaluation criteria
    eval_criteria = weak_instantiate_all(train_config['eval_criteria'])
    print('-- Evaluation Criteria :')
    if len(eval_criteria):
        for key, val in eval_criteria.items():
            print('  --', key, ':', val)
    else :
        print('  No evaluation criteria.')
        
    # Instantiate callbacks
    callbacks = weak_instantiate_all(train_config['callbacks'])
    callbacks_arguments = weak_instantiate_all(train_config['callbacks_arguments'])

    # print(callbacks_arguments)
    df = normalize_dataset(expand_df_labels(build_dataset()))
    full_dataset = SkinCancerDataset(
        df,
        input_cols      = model_config['input'],
        output_cols     = model_config['output'],
        input_img_dims  = model_config['input_img_dim'],
        path_cols       = model_config['path_cols'],
        extra_transforms= get_extra_transforms(0.2, *model_config['input_img_dim'][-2:]),
        flatten         = model_config['flatten'] if 'flatten' in model_config.keys() else False,
        return_key      = False,
        return_type     ='dict',
        seed            = train_config['seed']
    )
    groups = get_groups(
        exclude     = train_config['exclude_groups'] if 'exclude_groups' in train_config.keys() else None
    )
    print(groups.keys())
    train_loader, val_loader, *_ = smart_split_dataset(
        splits          = train_config['splits'],
        full_dataset    = full_dataset,
        groups          = groups,
        seed            = train_config['seed']
    )
    training_subdir = get_training_subdir(
        training_stage  = 1,
        model_hash      = model_config['hash'], 
        train_hash      = train_config['hash'],
        top_dir         = args.test_dir,
        test_version    = args.test_version,
    )
    try:
        model.cpu()
        os.makedirs(training_subdir, exist_ok=True)
        data = train_loader[0]
        
        if 'flatten' in model_config.keys() and model_config['flatten']:
            img = data[0].reshape(*model_config['input_img_dim'])
        else :
            img = data[0]
        
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(training_subdir,'data.png'), img.moveaxis(0,-1))
        
        if 'Segmentor' in data[1].keys():
            if 'flatten' in model_config.keys() and model_config['flatten']:
                img = data[1]['Segmentor'].reshape(-1,*model_config['input_img_dim'][-2:])
            else :
                img = data[1]['Segmentor']
                img = img.reshape(-1,*img.shape[-2:])
                
            plt.imsave(
                os.path.join(training_subdir,'targ_Segmentor.png'), 
                img.squeeze(0)
            )
        plt.close('all')
        data = data[0]
        print(
            '-- Model Summary :',
            get_summary(
                model,
                data,
                dest = os.path.join(training_subdir, 'summary'),
                depth = 10,
            )
        )
        tmp = load_dict(args.model_config) # avoid lambda overwrite
        tmp['Image Encoder Summary'] = str(get_summary(
            img_enc,
            data,
            depth = 10,
        ))
        data = img_enc(data.unsqueeze(0))
        # print('data', data.shape)
        for head_name, head in heads.items():
            tmp[f'{head_name} Summary'] = str(get_summary(
                head,
                data.squeeze(0),
                depth = 10,
            ))
            a = head(data)
            print(f'  -- {head_name} -- Range : ({a.min()}, {a.max()})')
            
        print('-- Updating model configuration file:', save_dict(tmp, args.model_config))
    except Exception as e:
        if 'tmp' in globals().keys():
            print('-- Updating model configuration file:', save_dict(tmp, args.model_config))
        print('-- Could not compute model summary due to the following error:')
        # print(e)
        raise e
        
    if 'sampler' in train_config.keys() and train_config['sampler']:
        classes = train_config['sampler']

        # Calculate class weights using the training set 
        train_labels = df.loc[full_dataset.get_keys(train_loader.indices),classes].apply(np.argmax, axis=1)
        val_labels   = df.loc[full_dataset.get_keys(val_loader.indices),  classes].apply(np.argmax, axis=1)

        class_sample_counts = np.unique_counts(train_labels)
        # class_weights = 1.0 / torch.tensor(class_sample_counts.counts, dtype=torch.float)
        class_weights = {
            int(key) : 1.0 / float(val)
                for key, val in zip(class_sample_counts.values, class_sample_counts.counts)
        }

        # print((1/class_weights ** 2).mean().sqrt())
        # print((1/class_weights).sqrt().mean() ** 2)

        # dataset_len = int(len(classes) * ((1/class_weights).sqrt().mean() ** 2))
        # dataset_len = int(len(classes) * (1/class_weights ** 2).mean().sqrt())
        dataset_len = int(len(class_weights) * 1. / np.mean(list(class_weights.values())))
        # dataset_len = int(len(class_weights) * (1/class_weights).mean())

        print(class_weights)
        tr_sampler = torch.utils.data.WeightedRandomSampler(
            weights     = train_labels.transform(lambda x: class_weights[x]).tolist(),
            num_samples = dataset_len,
            replacement = True,
        )
        val_sampler = torch.utils.data.WeightedRandomSampler(
            weights     = val_labels.transform(lambda x: class_weights[x]).tolist(),
            num_samples = len(val_labels),
            replacement = True,
        )
            
    train_loader = DataLoader(
        train_loader, **({
            'sampler'       : tr_sampler,
        } if tr_sampler in globals().keys() else {
            'shuffle'       : True,
        }),
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count(),
        persistent_workers  = True,
        pin_memory          = device == torch.device('cuda'),
    )
    val_loader = DataLoader(
        val_loader, **({
            'sampler'       : val_sampler,
        } if val_sampler in globals().keys() else {
            'shuffle'       : False,
        }),
        batch_size          = train_config['batch_size'],
        num_workers         = os.cpu_count(),
        # persistent_workers  = True,
        pin_memory          = device == torch.device('cuda'),
    )

    print('-- Using dataset split :', train_config['splits'])
    print('  -- Train      :', len(train_loader.dataset))
    print('  -- Validation :', len(val_loader.dataset))
    print('-- Sample weight:', train_config['sample_weight'])
    history = train(
        model,
        train_dataloader    = train_loader,
        eval_dataloader     = val_loader,
        criterion           = criterion,
        eval_criteria       = eval_criteria,
        optimizer           = optimizer,
        scheduler           = scheduler,
        epochs              = train_config['epochs'],
        patience            = train_config['patience'],
        sample_weight       = train_config['sample_weight'],
        update_limit        = False,
        top_dirname         = training_subdir,
        device              = device,
        evaluate_training   = False,
        saving_steps        = 1,
        show_pbar           = 'external',
        callbacks           = callbacks,
        callbacks_arguments = callbacks_arguments,
    )
    