import numpy as np
import torch
import os

def weight_transformer(weights: torch.Tensor, RSLT_CHANNELS: int, DATA_CHANNELS : int = 1, grids = None) -> torch.Tensor:
    if grids is None:
        grids = 1
        
    # Extend with zeros
    weights = weights.reshape(grids, -1, weights.shape[1])
    if weights.shape[1] % DATA_CHANNELS != 0:
        zeros = torch.zeros(grids, DATA_CHANNELS - weights.shape[1] % DATA_CHANNELS, weights.shape[2]).to(weights.dtype)
        weights = torch.cat([weights, zeros], dim=1)
    weights = weights.reshape(-1, weights.shape[-1])
    
    if weights.shape[1] % RSLT_CHANNELS != 0:
        zeros = torch.zeros(weights.shape[0], RSLT_CHANNELS - weights.shape[1] % RSLT_CHANNELS).to(weights.dtype)
        weights = torch.cat([weights, zeros], dim=1)
    
    # Split the weights along columns into RSLT_CHANNELS chunks
    split_rslt = torch.split(weights, RSLT_CHANNELS, dim=1)
    # example_shape = split_rslt[0].shape
    # print(f"After splitting along columns: torch.Size({list(example_shape)}) * {len(split_rslt)}")
    # print(f"First split_rslt tensor:\n{split_rslt[0]}")
    
    # Concatenate along rows
    concat_rows = torch.cat(split_rslt, dim=0)
    # print(f"After concatenating along rows: {concat_rows.shape}")

    return concat_rows

def save_tensor_to_bin(tensor, fname, verbose = False):
    np_array = np.asarray(tensor)
    np_array = np_array.astype(np_array.dtype.newbyteorder('<'))
    with open(fname, 'wb') as f:
        f.write(np_array.tobytes()) # default is 'C' order (row-major)
        
    if verbose:
        print(f"Saved tensor to {fname}")

def save_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1, DATA_CHANNELS : int = 1, grids = None):
    """
    Saves transformed linear layer weights to binary files (little endian).
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    weight_path = os.path.join(root_dir, "extracted_params",f"rslt_{RSLT_CHANNELS}_{DATA_CHANNELS}")
    os.makedirs(weight_path, exist_ok=True)
    
    for key in state_dict:
        # match = re.search(r'layers\.(\d+)', key)
        # if not match:
        if not key.endswith('.weight'):
            print(f"Key {key} does not contain layer index. Skipping.")
            continue
        # layer_idx = match.group(1)
        layer_idx = key.rstrip('.weight')
        
        # Extract weight tensor
        weight_tuple = state_dict[key]
        if isinstance(weight_tuple, tuple) :
            if len(weight_tuple) < 1:
                print(f"Invalid weight tuple for {key}. Skipping.")
                continue
            weight_tensor = weight_tuple[0]
        else :
            weight_tensor = weight_tuple
            
        if not isinstance(weight_tensor, torch.Tensor):
            print(f"Weight is not a tensor for {key}. Skipping.")
            continue

        if weight_tensor.dim() < 2:
            # 1-D tensors (e.g. LayerNorm scale/bias) are not linear weight
            # matrices and cannot be packetized — skip silently.
            continue

        # Convert quantized tensor to integer representation
        if weight_tensor.dtype == torch.qint8 and hasattr(weight_tensor, "int_repr"):
            weight_tensor = weight_tensor.int_repr().detach()
        else:
            weight_tensor = weight_tensor.detach().clone()
        
        if grids is None or isinstance(grids, int) :
            use_grid = grids
        elif isinstance(grids, dict) and f'{layer_idx}.grid' in grids.keys():
            use_grid = grids[f'{layer_idx}.grid'].numel()
        
        # Apply transformation
        try:
            transformed = weight_transformer(weight_tensor.T, RSLT_CHANNELS, DATA_CHANNELS, grids=use_grid)
        except Exception as e:
            print(f"Transformation failed for {key}: {e}")
            raise e
            continue
        
        if weight_tensor.dtype == torch.qint8 and hasattr(weight_tensor, "int_repr"):
            transformed = transformed.numpy().astype(np.int8)
        else:
            transformed = transformed.numpy()
            
        # filename = f"layer_{layer_idx}_weight.bin"
        filename = f"{key}.bin"
        filepath = os.path.join(weight_path, filename)
        save_tensor_to_bin(transformed, filepath)

def extract_fx_packed_params(state_dict):
    return {key: value for key, value in state_dict.items() 
            if '_packed_params._packed_params' in key}

def save_fx_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1, DATA_CHANNELS : int = 1):
    """
    Saves transformed linear layer weights to binary files (little endian).
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    packed_params = extract_fx_packed_params(state_dict)
    return save_weights_to_bin(packed_params, root_dir, RSLT_CHANNELS, DATA_CHANNELS)

def extract_custom_params(state_dict, parameter='weight'):
    # Support both direct state dicts and old-style wrapped dicts {'model_state_dict': {...}}
    inner = state_dict.get('model_state_dict', state_dict)
    return {key: value for key, value in inner.items()
            if f'.{parameter}' in key}

def save_custom_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1, DATA_CHANNELS : int = 1):
    """
    Saves transformed linear layer weights to binary files (little endian).
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    packed_params = extract_custom_params(state_dict)
    packed_grid_params = extract_custom_params(state_dict, 'grid')
    return save_weights_to_bin(packed_params, root_dir, RSLT_CHANNELS, DATA_CHANNELS, grids=packed_grid_params)

def save_custom_model_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1, DATA_CHANNELS : int = 1):
    """
    Saves transformed linear layer weights to binary files (little endian).
    
    Args:
        state_dict: Model's state dictionary.
        root_dir: Directory to save the binary files.
        RSLT_CHANNELS: number of results per iteration (if irrelevant, use default value)
    """
    save_custom_weights_to_bin(state_dict, root_dir, RSLT_CHANNELS, DATA_CHANNELS)
    param_dir = os.path.join(root_dir, "extracted_params")
    
    params = extract_custom_params(state_dict, 'grid')
    params.update(extract_custom_params(state_dict, 'inv_denom'))
    
    for key, val in params.items():
        save_tensor_to_bin(val, os.path.join(param_dir, f'{key}.bin'))
        # save_tensor_to_bin(val, os.path.join(param_dir, f'{key.replace('.','_')}.bin'))
        
def cumulate_files(filenames, src_dir, dest_fname, key):
    fnames = list(filter(lambda a: key in a, filenames))
    if len(fnames):
        with open(dest_fname, 'wb') as fw:
            for fname in fnames:
                with open(os.path.join(src_dir,fname), 'rb') as fr:
                    fw.write(fr.read())

        # print(f"Packetized {key} to '{dest_fname}'")
    
def packetize_model_to_bin(state_dict, root_dir, RSLT_CHANNELS = 1, DATA_CHANNELS : int = 1):
    save_custom_model_to_bin(state_dict, root_dir, RSLT_CHANNELS, DATA_CHANNELS)
    
    param_dir = os.path.join(root_dir, "extracted_params","")
    pckt_dir = os.path.join(root_dir, "packetized_params")
    
    os.makedirs(pckt_dir, exist_ok=True)
    
    for dirpath, dirnames, filenames in os.walk(param_dir):
        for key in ['weight','grid','inv_denom']:
            # Sort filenames
            filenames = [
                f'{param_key}.bin' for param_key in state_dict.keys()
                    if f'{param_key}.bin' in filenames
            ]
            cumulative_fname = dirpath.replace(param_dir,'')
            cumulative_fname = f'{cumulative_fname}_{key}.bin' if len(cumulative_fname) else f'{key}.bin'
            cumulative_fname = os.path.join(pckt_dir, cumulative_fname)
            
            cumulate_files(filenames, dirpath, cumulative_fname, key)
            
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Quantizer')
    parser.add_argument('-d','--data-channels', dest='data_channels', type=int, default=1, help='Parallel input channels (default: 1)')
    parser.add_argument('-r','--rslt-channels', dest='rslt_channels', type=int, default=1, help='Outputs per iteration (default: 1)')
    parser.add_argument('model_pth', type=str, help="The path to the target model's state dictionary")
    
    args = parser.parse_args()
    
    if os.path.exists(args.model_pth):
        state_dict = torch.load(args.model_pth)
        root_dir = os.path.dirname(args.model_pth)
        
        packetize_model_to_bin(state_dict, root_dir, args.rslt_channels, args.data_channels)