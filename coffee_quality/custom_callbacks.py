from typing import Iterable
import torch
import numpy as np
import pandas as pd
import os
import json
  
class ProbabilityAdjuster :
    def __init__(
        self,
        input : list = None,
        input_categories = None,
        output : list = None,
        output_categories = None,
        confusion_matrix : pd.DataFrame = None,
        metric_name = 'Accuracy',
        smoothing_coef = 0.75,
        log_dir : str = None,
        saving_interval : int = 1,
        saving_is_blocking : bool = True,
    ):
        # Translate input column names to positions
        self.input, self.input_categories, self.input_regression_type = \
            self._initialize(input, input_categories)
            
        # Translate output column names to positions
        self.output, self.output_categories, self.output_regression_type = \
            self._initialize(output, output_categories)
            
        # Initiate target metric & smoothing coefficient
        self.metric_name    = metric_name
        self.smoothing_coef = smoothing_coef
        
        # Transform confusion matrix
        # Build column lists using actual column names (not category base names)
        # For regression: use the column names directly
        # For categories: use all the expanded column names (e.g., Certification.Body_Is_AMECAFE)
        self.input_cols  = [*[self.input[_]  for _ in self.input_regression_type]]
        for cat_indices in self.input_categories.values():
            self.input_cols.extend([self.input[idx] for idx in cat_indices])
            
        self.output_cols = [*[self.output[_] for _ in self.output_regression_type]]
        for cat_indices in self.output_categories.values():
            self.output_cols.extend([self.output[idx] for idx in cat_indices])

        if confusion_matrix is None :
            self.cm = torch.ones(len(self.input_cols), len(self.output_cols)) / len(self.input_cols)
            common = np.asarray(self.input_cols)[np.isin(self.input_cols, self.output_cols)]
            couples = ((self.input_cols.index(_),self.output_cols.index(_)) for _ in common)
            
            for idx_i, cols_i in couples:
                self.cm[idx_i,cols_i] = 1.
        else :
            self.cm = torch.tensor(
                pd.DataFrame(confusion_matrix).loc[pd.Index(self.input_cols), self.output_cols].values, 
                requires_grad=False
            )
            
        self.cm = self.cm.float().abs() ** 2
        
        self.logs = None if log_dir is None else os.path.join(log_dir, 'ProbabilityAdjuster.json')
        if self.logs is not None:            
            if os.path.exists(self.logs):
                with open(self.logs, 'r') as fr:
                    self.prob_dict = json.load(fr)
            else :
                self.prob_dict = {
                    'input_probabilities' : {
                        label : {} for label in self.input_cols
                    }, 
                    'output_probabilities' : {
                        label : {} for label in self.output_cols
                    },
                }
            self.saving_interval = saving_interval
            self.saving_counter = saving_interval
            
            self.saving_is_blocking = saving_is_blocking
            if not self.saving_is_blocking:
                self.mp = None
        
        # Initiate probabilities
        # Probabilities are per "component": one per regression column + one per category group
        self.num_input_components = len(self.input_regression_type) + len(self.input_categories)
        self.num_output_components = len(self.output_regression_type) + len(self.output_categories)
        
        self._input_prob  = torch.ones(self.num_input_components,  requires_grad=False).float()
        self._output_prob = torch.ones(self.num_output_components, requires_grad=False).float()
        
    @classmethod  
    def _initialize(cls, columns, categories) :
        if categories is None:
            categories = []
        columns = {
            key : idx for idx, key in enumerate(columns) 
        }
        categories = {
            category[0][:category[0].find('_Is_')] : [
                columns[key] for key in category
            ] for category in categories
                if len(category) > 0
        }
        regression_type = columns.values()
        
        for category in categories.values():
            regression_type = [
                idx for idx in regression_type
                    if idx not in category
            ]
        columns = {idx : key for key, idx in columns.items()}
        return columns, categories, regression_type
    
    @classmethod  
    def _to_prob(cls, logits):
        # logits[logits < 0.] = 0.
        # logits[logits > 1.] = 1.
        # return logits ** 2
        return logits.abs() * torch.sigmoid(logits)
    
    @classmethod  
    def _expand_prob(cls, prob, columns, categories, regression_type):
        target_prob = torch.empty(len(columns), dtype=prob.dtype, device=prob.device)
        
        # Copy regression-type probabilities
        regression_indices = list(regression_type)
        target_prob[regression_indices] = prob[:len(regression_indices)]
        
        # Copy probabilities for each category
        for offset, category_indices in enumerate(categories.values()):
            target_prob[list(category_indices)] = prob[len(regression_indices)+offset]
        
        return target_prob
    
    def _backprop_prob(self, out_prob):
        # Expand component probabilities to full column space for confusion matrix
        expanded_out = self._expand_prob(
            out_prob,
            self.output,
            self.output_categories,
            self.output_regression_type
        )
        
        # Backprop through confusion matrix
        expanded_in = (expanded_out @ self.cm.T) / self.cm.sum(1)
        
        # Compress back to component space
        in_prob = torch.ones(self.num_input_components, dtype=expanded_in.dtype, device=expanded_in.device)
        
        # Regression components: direct copy
        for idx, col_idx in enumerate(self.input_regression_type):
            in_prob[idx] = expanded_in[col_idx]
        
        # NOTE: Category components: average over the category group ?
        for idx, category_indices in enumerate(self.input_categories.values()):
            in_prob[len(self.input_regression_type) + idx] = \
                torch.stack([expanded_in[col_idx] for col_idx in category_indices]).mean()
        
        return in_prob
    
    def _smooth(self, x, x_smoothed):
        return self.smoothing_coef * x + (1-self.smoothing_coef) * x_smoothed
    
    def _update_logs(self, epoch, in_prob, out_prob):
        if self.logs is not None:
            # Expand probabilities to column form for logging
            expanded_in = self._expand_prob(
                torch.tensor(in_prob),
                self.input,
                self.input_categories,
                self.input_regression_type
            )
            expanded_out = self._expand_prob(
                torch.tensor(out_prob),
                self.output,
                self.output_categories,
                self.output_regression_type
            )
            
            # Add input probabilities
            for idx, val in enumerate(expanded_in):
                self.prob_dict['input_probabilities'][self.input_cols[idx]][epoch] = float(val)
                
            # Add output probabilities
            for idx, val in enumerate(expanded_out):
                self.prob_dict['output_probabilities'][self.output_cols[idx]][epoch] = float(val)
                
            self.saving_counter -= 1
            if self.saving_counter == 0:
                self.save_logs(timeout=10)
                self.saving_counter = self.saving_interval
                
                
    def __save_logs(self, prob_dict):
        os.makedirs(os.path.dirname(self.logs), exist_ok=True)
        with open(self.logs, 'w') as fw:
            json.dump(prob_dict, fw, indent=2)
                
    def save_logs(self, timeout=None):
        if self.logs is not None:
            if not self.saving_is_blocking:
                if self.mp is not None:
                    self.mp.join(timeout)
                    if timeout is not None and self.mp.is_alive():
                        print("Warning: Previous saving process is still running and timeout reached!")
                        return
                assert self.mp is None or not self.mp.is_alive(), "Previous saving process is still running!"    
                self.mp = torch.multiprocessing.get_context('spawn').Process(
                    target=self.__save_logs,
                    args=(self.prob_dict.copy(),)
                )
                self.mp.start()
            else :
                self.__save_logs(self.prob_dict.copy())
                
    def update(
        self,
        history : dict[str, list | float] = None,
        epoch  = 0,
        **kwargs
    ):
        with torch.no_grad():
            acc = history['val'][epoch][self.metric_name]
            if isinstance(acc, dict):
                acc = list(acc.values())
            out_prob = self._to_prob(torch.tensor(acc).float())
            
            self._output_prob = self._smooth(out_prob, self._output_prob)
            in_prob = self._backprop_prob(out_prob)
            
            self._input_prob = self._smooth(in_prob, self._input_prob)
            self._update_logs(epoch, in_prob.tolist(), out_prob.tolist())
        
    def get_input_prob(self, *args, expanded = False, **kwargs):
        if expanded:
            return self._expand_prob(
                self._input_prob.detach().clone(),
                self.input,
                self.input_categories,
                self.input_regression_type,
            )
        else :
            return self._input_prob.detach().clone()
    
    def get_output_prob(self, *args, expanded = False, **kwargs):
        if expanded:
            return self._expand_prob(
                self._output_prob.detach().clone(),
                self.output,
                self.output_categories,
                self.output_regression_type,
            )
        else :
            return self._output_prob.detach().clone()
    
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        
class MaskInput :
    def __init__(
        self, 
        input : list = None,
        input_categories = None,
        max_probability = 0.25,
        x_shift = 0.,
        masked_value = -1,
    ):
        self.max_probability = max_probability
        self.x_shift = x_shift
        self.masked_value = torch.tensor(masked_value)
        
        self.input = input
        self.input_categories = input_categories
        
        if self.input_categories is None:
            self.input_categories = []
            
        if self.input is None:
            self.input_categories = []
            
        else :
            self._initialize_input(input)
    
    def _initialize_input(self, input) :
            self.input = {
                key : idx for idx, key in enumerate(self.input) 
            }
            self.input_categories = [[
                    self.input[key] for key in category
                ] for category in self.input_categories
            ]
            self.input_regression_type = self.input.values()
            
            for category in self.input_categories:
                self.input_regression_type = [
                    idx for idx in self.input_regression_type
                        if idx not in category
                ] 
            self._initialize_input = lambda input : None
            
    def __call__(
        self,
        data        : torch.Tensor, 
        epoch       : int,
        epochs      : int,
        dataloader  : Iterable,
        iteration   : int = None,
        device      = torch.device('cpu'),
        probability_adjuster : ProbabilityAdjuster = None,
        **kwargs
    ) :
        self._initialize_input(data.shape[-1])
        if iteration is None:
            iteration = len(dataloader)
            
        probability = self.max_probability / (1 + np.exp((epoch / epochs) * (iteration / len(dataloader)) - self.x_shift))
        
        if probability_adjuster is not None:
            probability *= probability_adjuster.get_input_prob(expanded=True)
        probability = torch.as_tensor(probability, device = device)
        
        mask = torch.rand(len(self.input_regression_type) + len(self.input_categories), device = device)
        
        mask_extended = torch.empty(len(self.input), device = device, dtype = mask.dtype)
        regression_indices = list(self.input_regression_type)
        mask_extended[regression_indices] = mask[:len(regression_indices)]
        
        for offset, category in enumerate(self.input_categories):
            mask_extended[category] = mask[len(self.input_regression_type) + offset]
            
        mask_extended = mask_extended < probability
        
        data.masked_fill_(mask_extended, self.masked_value.to(data.dtype).to(device))
      