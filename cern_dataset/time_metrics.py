#!/usr/bin/env python3
if __name__ == '__main__':
    import sys, os
    from argparse import ArgumentParser

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    TOP_DIR  = os.path.dirname(THIS_DIR)
    sys.path.append(TOP_DIR)

    parser = ArgumentParser(description='Quantize trained FasterKAN models for the CERN dataset.')
    parser.add_argument('-d', '--train-dir', dest='train_dir', default=os.path.join(THIS_DIR, 'train'), help='Top-level training output directory (default: ./train)')
    parser.add_argument('-l', '--limit', dest='limit', type=int, default=-1, help='Limit the number of characters of the hashes shown in the figures.')
    parser.add_argument('--epoch', dest='epoch', type=str, default='best')

    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--all', dest='all_models', action='store_true', default=False, help='Process all models in the train directory (default behaviour)')
    sel.add_argument('-m','--model', dest='model_name', type=str, default=None, help='Process a specific run by folder name, e.g. test_0 or <hash>/test_0')

    args = parser.parse_args()

    # ------------------------------------------------------------------ 
    import torch
    import pandas as pd 
    from torch.utils.data import DataLoader
    import torch.utils.benchmark as benchmark

    from kan_utils.config.config import load_config, instantiate
    from kan_utils.utils import load_model, save_model, save_dict
    from kan_utils.quantization import FixedPointFasterKAN, FloatWrapperModule
    
    from prepare_dataset import build_dataset, get_dataset_paths, normalize_data
    from custom_dataset import DistributedH5Dataset

    device = torch.device("cpu") 

    model_pths= []
    top_search_path = args.train_dir
    if args.model_name is not None:
        top_search_path = os.path.join(
            top_search_path, 
            args.model_name,
        )
        if not os.path.exists(top_search_path):
            raise ValueError(f'Provided model does not exist inside training directory "{args.train_dir}"')
        
    for root, dirs, files in os.walk(top_search_path):
        for file in files:
            if args.epoch in file and os.path.splitext(file)[1]  in ('.pth','.pt'):
                model_pths.append(os.path.join(root, file))

    build_dataset() 
    test_dataset = get_dataset_paths('test')
    
    def model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        size = os.path.getsize("tmp.pt")
        os.remove("tmp.pt")
        return size

    # Compare takes a list of measurements which we'll save in results.
    results = []
    model_sizes = []
    for model_pth in model_pths:
        # Locate configuration files
        model_top_dir = os.path.dirname(model_pth)
        while 'config' not in os.listdir(model_top_dir):
            model_top_dir = os.path.dirname(model_top_dir)
        
        model_hash    = os.path.basename(os.path.dirname(model_top_dir))
        model_version = os.path.basename(model_top_dir)
        
        model_config = load_config(os.path.join(model_top_dir, 'config', 'model.json'))
        train_config = load_config(os.path.join(model_top_dir, 'config', 'train.json'))
        
        base_model   = instantiate(model_config, 'model')
        
        test_loader = DistributedH5Dataset(
            h5files               = test_dataset,
            buffer_size           = 2048,
            input_cols            = model_config['input'],
            output_cols           = model_config['output'],
            key_col               = None,
            task                  = train_config['task'],
            remove_mass_pt_window = model_config['remove_mass_pt_window'],
            preprocess_data       = lambda data, features = model_config['input']: normalize_data(data, features),
            preprocess_targ       = None,
        )
        test_loader = DataLoader(
            test_loader, 
            batch_size      = 1,
            num_workers     = 1,
            pin_memory      = device == torch.device('cuda'),
        )
        if (os.sep + 'models') in model_pth:
            model = base_model
            model_dr = 'fp32'
        elif (os.sep + 'quantized_models') in model_pth:
            if isinstance(base_model, torch.nn.Sequential) and len(base_model) == 1:
                base_model = next(base_model.children())
            model = FixedPointFasterKAN(base_model)
            model_dr = os.path.basename(os.path.dirname(model_pth))
        # print('-- Model :', model)
        print('-- Benchmarking', model_hash[:args.limit], model_version, model_dr)
            
        load_model(model, model_pth, device)

        if isinstance(model, FixedPointFasterKAN):
            model = FloatWrapperModule(model)
            
        # label and sub_label are the rows
        # description is the column
        model_results = []
        for num_threads in range(1,torch.get_num_threads()+1):
            print('  -- Threads :', num_threads)
            model_results.append(
                benchmark.Timer(
                    stmt        = 'model(data)',
                    globals     = {'model': model, 'data' : next(iter(test_loader))[0]},
                    num_threads = num_threads,
                    label       = f'{model_hash}',
                    sub_label   = model_dr,
                    env         = model_version,
                ).blocked_autorange(min_run_time=10)
            )
        model_sizes.append({
            'Hash'      : model_hash,
            'Version'   : model_version,
            'Data_repr' : model_dr,
            'Size'      : model_size(model),
            'Latency'   : benchmark.Compare(model_results)
        })
        results.extend(model_results)

    compare = benchmark.Compare(results)

    # Evaluate model sizes
    f_rslt = os.path.join(args.train_dir,'comparison',f'mem_results_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.txt')
    model_sizes = pd.DataFrame.from_records(model_sizes, exclude=['Latency'])
    model_sizes.to_csv(f_rslt, index=False)
    print(f'Memory sizes saved in {f_rslt}')
    
    print(results[0])

    tmp = sys.stdout
    f_rslt = os.path.join(args.train_dir,'comparison',f'time_results_{pd.Timestamp.now().strftime("%Y-%m-%d-%X")}.txt')
    
    with open(f_rslt, 'w') as sys.stdout:
        compare.print()
        
    sys.stdout = tmp
    print(f'Results saved in {f_rslt}')