from typing import overload
import os
import torch
from torchinfo import summary, ModelStatistics

@overload
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest = None,
    depth : int = 3,
) -> ModelStatistics :
    """
    Saves the attributes and details of a PyTorch model to a text file, including
    its parameter counts, architecture summary, and MACs (Multiply-Accumulate Operations).
    Also creates a checkpoint directory using the updated checkpoint module logic.

    Parameters
    ----------
    model : torch.nn.Module 
        The model to summarize. Model will be casted to evaluation mode.
    dataset : torch.Tensor
        A dataset containing input
    depth : int
        The maximum allowed depth of nested layers to report.

    Returns
    -------
    ModelStatistics
        The model summary.
    """
    ...

@overload
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest : str,
    depth : int = 3,
) -> str:
    """
    Saves the attributes and details of a PyTorch model to a text file, including
    its parameter counts, architecture summary, and MACs (Multiply-Accumulate Operations).
    Also creates a checkpoint directory using the updated checkpoint module logic.

    Parameters
    ----------
    model : torch.nn.Module 
        The model to summarize. Model will be casted to evaluation mode.
    dataset : torch.Tensor
        A dataset containing input
    dest: str
        The path to save the model summary.
    depth : int
        The maximum allowed depth of nested layers to report.

    Returns
    -------
    dest: str
        The path that the model summary was saved.
    """
    ...
    
def get_summary(
    model: torch.nn.Module,
    input_data : torch.Tensor,
    dest = None,
    depth : int = 3,
):
    model_summary = summary(
        model, 
        input_data  = input_data.clone().unsqueeze(0), 
        verbose     = 0,
        depth       = depth
    )
    
    if dest is None:
        return model_summary
    
    if os.path.splitext(dest)[-1] != '.txt':
        dest += '.txt'
    
    with open(dest, 'w') as fw:
        fw.write(str(model_summary))

    return dest