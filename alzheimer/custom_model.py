import torch
from warnings import warn
from kan_utils.models import FasterKAN, FasterKANLayer, DynamicFasterKANLayer, SubBatch, Reshaper
from kan_utils.utils import expand_value

class ImageKANEncoder(torch.nn.Module):
    '''Image encoder model using FasterKAN layers.
    
    :param input_shape: Shape of the input data (height, width).
    :type input_shape: tuple[int,int]
    :param hidden_state_size: Size of the hidden state in the first FasterKAN layer.
    :type hidden_state_size: int, list[int]
    :param encoded_state_size: Size of the encoded state in the second FasterKAN layer.
    :type encoded_state_size: int, list[int]
    :param num_grids: Number of grids for the FasterKAN layers.
    :type num_grids: int
    :param grid_min: Minimum grid values for the FasterKAN layers.
    :type grid_min: float
    :param grid_max: Maximum grid values for the FasterKAN layers.
    :type grid_max: float
    :param inv_denominator: Inverse denominator values for the FasterKAN layers.
    :type inv_denominator: float
    :param mode: Mode for the FasterKAN layers.
    :type mode: str
    :param residual: Whether to use residual connections in the FasterKAN layers.
    :type residual: bool
    :param dynamic: Whether to use dynamic behavior in the FasterKAN layers.
    :type dynamic: bool
    :param dropout_rate: Dropout rate for the FasterKAN layers.
    :type dropout_rate: float
    '''
    def __init__(
        self, 
        input_shape         , 
        hidden_state_size   ,
        encoded_state_size  ,
        num_grids           ,
        grid_min            ,
        grid_max            ,
        inv_denominator     ,
        mode                ,
        residual            = False,
        dynamic             = False,
        dropout_rate        = 0.5,
    ):
        super(ImageKANEncoder, self).__init__()
        
        if not hasattr(hidden_state_size, '__iter__'):
            hidden_state_size = [hidden_state_size]
        if not hasattr(encoded_state_size, '__iter__'):
            encoded_state_size = [encoded_state_size]
        
        self.row_enc = SubBatch(
            input_data_dim  =-1, 
            output_data_dim =-1, 
            model           = FasterKAN(
                hidden_layers   =[input_shape[-1], *hidden_state_size],
                num_grids       = num_grids,
                grid_min        = grid_min,
                grid_max        = grid_max,
                inv_denominator = inv_denominator,
                mode            = mode,
                residual        = residual,
                dynamic         = dynamic,
                dropout_rate    = dropout_rate,
            )
        )
        self.flatten = torch.nn.Flatten(start_dim=-2)
        self.col_enc = FasterKAN(
            hidden_layers   =[input_shape[0] * hidden_state_size[-1], *encoded_state_size],
            num_grids       = num_grids,
            grid_min        = grid_min,
            grid_max        = grid_max,
            inv_denominator = inv_denominator,
            mode            = mode,
            residual        = residual,
            dynamic         = dynamic,
            dropout_rate    = dropout_rate,
        )

    def forward(self, x):
        x = self.row_enc(x)
        x = self.flatten(x)
        x = self.col_enc(x)
        return x
    
class ImageKANDecoder(torch.nn.Module):
    '''Image decoder model using FasterKAN layers.
    
    :param output_shape: Shape of the output data (height, width).
    :type output_shape: tuple[int,int]
    :param hidden_state_size: Size of the hidden state in the first FasterKAN layer.
    :type hidden_state_size: int, list[int]
    :param encoded_state_size: Size of the encoded state in the second FasterKAN layer.
    :type encoded_state_size: int, list[int]
    :param num_grids: Number of grids for the FasterKAN layers.
    :type num_grids: int
    :param grid_min: Minimum grid values for the FasterKAN layers.
    :type grid_min: float
    :param grid_max: Maximum grid values for the FasterKAN layers.
    :type grid_max: float
    :param inv_denominator: Inverse denominator values for the FasterKAN layers.
    :type inv_denominator: float
    :param mode: Mode for the FasterKAN layers.
    :type mode: str
    :param residual: Whether to use residual connections in the FasterKAN layers.
    :type residual: bool
    :param dynamic: Whether to use dynamic behavior in the FasterKAN layers.
    :type dynamic: bool
    :param dropout_rate: Dropout rate for the FasterKAN layers.
    :type dropout_rate: float
    '''
    def __init__(
        self, 
        output_shape        , 
        hidden_state_size   ,
        encoded_state_size  ,
        num_grids           ,
        grid_min            ,
        grid_max            ,
        inv_denominator     ,
        mode                ,
        residual            = False,
        dynamic             = False,
        dropout_rate        = 0.5,
    ):
        super(ImageKANDecoder, self).__init__()
        
        if not hasattr(hidden_state_size, '__iter__'):
            hidden_state_size = [hidden_state_size]
        if not hasattr(encoded_state_size, '__iter__'):
            encoded_state_size = [encoded_state_size]
        
        self.col_dec = FasterKAN(
            hidden_layers   = [*encoded_state_size, output_shape[0] * hidden_state_size[0]],
            num_grids       = num_grids,
            grid_min        = grid_min,
            grid_max        = grid_max,
            inv_denominator = inv_denominator,
            mode            = mode,
            residual        = residual,
            dynamic         = dynamic,
            dropout_rate    = dropout_rate,
        )
        self.reshaper = Reshaper(
            input_data_shape  = [output_shape[0] * hidden_state_size[0],],
            output_data_shape = [output_shape[0] , hidden_state_size[0],],
        )
        self.row_dec = SubBatch(
            input_data_dim  =-1, 
            output_data_dim =-1,
            model           = FasterKAN(
                hidden_layers   = [*hidden_state_size, output_shape[-1]],
                num_grids       = num_grids,
                grid_min        = grid_min,
                grid_max        = grid_max,
                inv_denominator = inv_denominator,
                mode            = mode,
                residual        = residual,
                dynamic         = dynamic,
                dropout_rate    = dropout_rate,
            )
        )

    def forward(self, x):
        x = self.col_dec(x)
        x = self.reshaper(x)
        x = self.row_dec(x)
        return x
    
class RecurrentKANEncoder(torch.nn.Module): 
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_layers, 
        num_grids, 
        grid_min, 
        grid_max, 
        inv_denominator, 
        mode, 
        residual            = False, 
        dynamic             = False, 
        dropout_rate        = 0.5,
        return_sequence     = False,
        return_stack        = False,
    ):
        super(RecurrentKANEncoder, self).__init__()
        
        if not hasattr(hidden_layers, '__iter__'):
            hidden_layers = [hidden_layers]
            
        hidden_layers       = [input_size, *hidden_layers, output_size]
        self._input_sizes   = hidden_layers[:-1]
        self.input_sizes    = [a + b for a, b in zip(hidden_layers[:-1], hidden_layers[1:])]
        self.output_sizes   = hidden_layers[1:]
        
        num_grids       = expand_value(num_grids,       len(self.input_sizes))
        grid_min        = expand_value(grid_min,        len(self.input_sizes))
        grid_max        = expand_value(grid_max,        len(self.input_sizes))
        inv_denominator = expand_value(inv_denominator, len(self.input_sizes))
        residual        = expand_value(residual,        len(self.input_sizes))
        
        if dynamic :
            LayerClass = DynamicFasterKANLayer
        else :
            LayerClass = FasterKANLayer
        
        self.residual   = []
        for _iter, residual_i in enumerate(residual):
            if residual_i :
                if hidden_layers[_iter] == hidden_layers[_iter+1] :
                    self.residual.append(True)
                else :
                    warn(f"Skipped residual connection at layer {_iter}; Number of features do not match ({hidden_layers[_iter]} != {hidden_layers[_iter+1]})")
                    self.residual.append(False)
            else :
                self.residual.append(False)
        
        self.layers = torch.nn.ModuleList([
            LayerClass(
                input_dim      = self.input_sizes[_iter],
                output_dim     = self.output_sizes[_iter],
                num_grids       = num_grids[_iter],
                grid_min        = grid_min[_iter],
                grid_max        = grid_max[_iter],
                inv_denominator = inv_denominator[_iter],
                mode            = mode,
                dropout_rate    = dropout_rate,
            )
            for _iter in range(len(self.input_sizes))
        ])
        self.init_state = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.empty(output_size),
                    a = -(1./output_size)**0.5,
                    b =  (1./output_size)**0.5,
                ))
                for output_size in self.output_sizes
        ])
        self.return_sequence = return_sequence
        self.return_stack    = return_stack
    
    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"residual={self.residual}, return_sequence={self.return_sequence}, return_stack={self.return_stack}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_size, output_dim]
        """
        out = [
            torch.empty(
                (*x.shape[:-1], output_size), 
                dtype   = x.dtype,
                device  = x.device,
            ) 
                for output_size in self.output_sizes
        ]
        x_in = [
            torch.empty(
                (*x.shape[:-2], input_sizes),
                dtype   = x.dtype,
                device  = x.device,
            ) 
                for input_sizes in self.input_sizes
        ]
        for _iter, x_i in enumerate(x.unbind(dim=-2)):
            for layer_i, (layer, res) in enumerate(zip(self.layers, self.residual)):
                # if layer_i == 0:
                #     x_in = x_i
                # else :
                #     x_in = out[layer_i-1][..., _iter, :]
                    
                # if _iter == 0 :
                #     x_in = torch.concat([
                #         x_in, 
                #         self.init_state[layer_i].expand(*(*x_in.shape[:-1], layer.output_dim))
                #     ], dim=-1)
                # else : 
                #     x_in = torch.concat([x_in, out[layer_i][..., _iter-1, :]], dim=-1)
                    
                # if res:
                #     x_in = layer(x_in) + x_in[...,:layer.output_dim]
                # else :
                #     x_in = layer(x_in)
                    
                # out[layer_i][..., _iter, :] = x_in
                
                x_in[layer_i][...,:self._input_sizes[layer_i]] = (
                    out[layer_i-1][..., _iter, :]           if layer_i > 0 else
                    x_i
                )
                x_in[layer_i][...,self._input_sizes[layer_i]:] = (
                    out[layer_i][..., _iter-1, :]           if _iter   > 0 else 
                    self.init_state[layer_i]
                )
                out[layer_i][..., _iter, :] = layer(x_in[layer_i]) 
                if res:
                    out[layer_i][..., _iter, :] += x_in[layer_i][...,:self._input_sizes[layer_i]]
                
        if self.return_stack and self.return_sequence :
            return out
        elif self.return_sequence :
            return out[-1]
        elif self.return_stack :
            return [out_i[..., -1, :] for out_i in out]
        else :
            return out[-1][..., -1, :]
        
         
class RecurrentKANDecoder(torch.nn.Module): 
    def __init__(
        self, 
        input_size, 
        output_size, 
        hidden_layers, 
        num_grids, 
        grid_min, 
        grid_max, 
        inv_denominator, 
        mode, 
        residual            = False, 
        dynamic             = False, 
        dropout_rate        = 0.5,
        return_sequence     = False,
        return_stack        = False,
        sequence_length     = 1,
    ):
        super(RecurrentKANDecoder, self).__init__()
        
        if not hasattr(hidden_layers, '__iter__'):
            hidden_layers = [hidden_layers]
            
        hidden_layers       = [input_size, *hidden_layers, output_size]
        self._input_sizes   = hidden_layers[:-1]
        self.input_sizes    = [a + b for a, b in zip(hidden_layers[:-1], hidden_layers[1:])]
        self.output_sizes   = hidden_layers[1:]
        
        num_grids       = expand_value(num_grids,       len(self.input_sizes))
        grid_min        = expand_value(grid_min,        len(self.input_sizes))
        grid_max        = expand_value(grid_max,        len(self.input_sizes))
        inv_denominator = expand_value(inv_denominator, len(self.input_sizes))
        residual        = expand_value(residual,        len(self.input_sizes))
        
        if dynamic :
            LayerClass = DynamicFasterKANLayer
        else :
            LayerClass = FasterKANLayer
        
        self.residual   = []
        for _iter, residual_i in enumerate(residual):
            if residual_i :
                if hidden_layers[_iter] == hidden_layers[_iter+1] :
                    self.residual.append(True)
                else :
                    warn(f"Skipped residual connection at layer {_iter}; Number of features do not match ({hidden_layers[_iter]} != {hidden_layers[_iter+1]})")
                    self.residual.append(False)
            else :
                self.residual.append(False)
        
        self.layers = torch.nn.ModuleList([
            LayerClass(
                input_dim       = self.input_sizes[_iter],
                output_dim      = self.output_sizes[_iter],
                num_grids       = num_grids[_iter],
                grid_min        = grid_min[_iter],
                grid_max        = grid_max[_iter],
                inv_denominator = inv_denominator[_iter],
                mode            = mode,
                dropout_rate    = dropout_rate,
            )
            for _iter in range(len(self.input_sizes))
        ])
        self.feedback = torch.nn.Identity() if input_size == output_size else SubBatch(
            input_data_dim  = -1,
            output_data_dim = -1,
            model           = LayerClass(
                input_dim       = output_size,
                output_dim      = input_size,
                num_grids       = num_grids[0],
                grid_min        = grid_min[0],
                grid_max        = grid_max[0],
                inv_denominator = inv_denominator[0],
                mode            = mode,
                dropout_rate    = dropout_rate,
            )
        )
        self.init_state = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.empty(output_size),
                    a = -(1./output_size)**0.5,
                    b =  (1./output_size)**0.5,
                ))
                for output_size in self.output_sizes
        ])
        self.return_sequence = return_sequence
        self.return_stack    = return_stack
        self.sequence_length = sequence_length
    
    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"residual={self.residual}, return_sequence={self.return_sequence}, return_stack={self.return_stack}, sequence_length={self.sequence_length}"

    def forward(self, x: torch.Tensor, sequence_length=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_size, output_dim]
        """
        if sequence_length is None :
            sequence_length = self.sequence_length
        
        out = [
            torch.empty(
                (*x.shape[:-1], sequence_length, output_size),
                dtype   = x.dtype,
                device  = x.device,
            ) 
                for output_size in self.output_sizes
        ]
        x_in = [
            torch.empty(
                (*x.shape[:-1], input_sizes),
                dtype   = x.dtype,
                device  = x.device,
            ) 
                for input_sizes in self.input_sizes
        ]
        for _iter in range(sequence_length):
            # if _iter == 0 :
            #     x_i = x
            # else :
            #     x_i = self.feedback(out[-1][..., _iter-1, :])
            
            for layer_i, (layer, res) in enumerate(zip(self.layers, self.residual)):
                # if layer_i == 0:
                #     x_in = x_i
                # else :
                #     x_in = out[layer_i-1][..., _iter, :]
                
                # if _iter == 0 :
                #     x_in = torch.concat([
                #         x_in, 
                #         self.init_state[layer_i].expand(*x_in.shape[:-1], -1)
                #     ], dim=-1)
                # else : 
                #     x_in = torch.concat([x_in, out[layer_i][..., _iter-1, :]], dim=-1)
                    
                # if res:
                #     x_in = layer(x_in) + x_in[...,:layer.output_dim]
                # else :
                #     x_in = layer(x_in)
                    
                # out[layer_i][..., _iter, :] = x_in
                    
                x_in[layer_i][..., :-self.output_sizes[layer_i]] = (
                    out[layer_i-1][..., _iter, :]           if layer_i > 0 else
                    self.feedback(out[-1][..., _iter-1, :]) if _iter   > 0 else 
                    x
                )
                x_in[layer_i][..., -self.output_sizes[layer_i]:] = (
                    out[layer_i][..., _iter-1, :]           if _iter   > 0 else 
                    self.init_state[layer_i].broadcast_to(
                        *out[layer_i][..., _iter-1, :].shape[:-1],
                        self.output_sizes[layer_i]
                    )
                )
                # x_in[layer_i][..., :] = torch.cat([(
                #             out[layer_i-1][..., _iter, :]           if layer_i > 0 else
                #             self.feedback(out[-1][..., _iter-1, :]) if _iter   > 0 else 
                #             x
                #         ),(
                #             out[layer_i][..., _iter-1, :]           if _iter   > 0 else 
                #             self.init_state[layer_i].broadcast_to(
                #                 *out[layer_i][..., _iter-1, :].shape[:-1],
                #                 self.output_sizes[layer_i]
                #             )
                #     )],
                #     dim = -1
                # )
                out[layer_i][..., _iter, :] = layer(x_in[layer_i]) 
                if res:
                    out[layer_i][..., _iter, :] += x_in[layer_i][...,:layer.output_dim]
                
        if self.return_stack and self.return_sequence :
            return out
        elif self.return_sequence :
            return out[-1]
        elif self.return_stack :
            return [out_i[..., -1, :] for out_i in out]
        else :
            return out[-1][..., -1, :]
        

class LSTMEncoder(torch.nn.LSTM):
    def __init__(
        self, 
        *args, 
        return_sequence     = False, 
        return_states       = False, 
        trainable_states    = False,
        **kwargs
    ):
        super(LSTMEncoder,self).__init__(*args,**kwargs)
        self.return_sequence = return_sequence
        self.return_states   = return_states
        self.D = 1 + int(self.bidirectional)
        self.output_size = (self.proj_size if self.proj_size > 0 else self.hidden_size)
        
        layer_len, size = self.D *self.num_layers, self.hidden_size
        self.c_0 = torch.zeros(layer_len, size)
        
        layer_len, size = self.D * self.num_layers, self.output_size
        self.h_0 = torch.zeros(layer_len, size)
        
        if trainable_states:
            self.c_0 = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    self.c_0,
                    a =-(1. / self.c_0.numel()) ** 0.5,
                    b = (1. / self.c_0.numel()) ** 0.5,
                )
            )
            self.h_0 = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    self.h_0,
                    a =-(1. / self.h_0.numel()) ** 0.5,
                    b = (1. / self.h_0.numel()) ** 0.5,
                )
            )
            
        
    def forward(self, x):
        if len(x.shape) == 3:
            h_0 = torch.stack([self.h_0 for _ in range(x.shape[1-self.batch_first])], dim=1)
            c_0 = torch.stack([self.c_0 for _ in range(x.shape[1-self.batch_first])], dim=1)
        else :
            h_0, c_0 = self.h_0, self.c_0
            
        x, (h_n, c_n) = super().forward(x, (h_0, c_0))
        
        if self.return_sequence and self.return_states:
            return x, (h_n, c_n)
        elif self.return_sequence:
            return x
        elif self.return_states:
            return (h_n, c_n)
        else :
            if self.batch_first == False or len(x.shape) == 2:
                return x[-1]
            else :
                return x[...,-1,:]
        
class LSTMDecoder(torch.nn.LSTM):
    def __init__(
        self, 
        *args, 
        return_sequence     = False, 
        return_states       = False, 
        trainable_states    = False,
        seq_len             = None,
        feedback            = None,
        **kwargs
    ):
        self._batch_first = kwargs['batch_first']
        kwargs['bidirectional'] = False
        kwargs['batch_first']  = False
        super(LSTMDecoder,self).__init__(*args,**kwargs)
        self.return_sequence = return_sequence
        self.return_states   = return_states
        self.D = 1 + int(self.bidirectional)
        self.output_size = (self.proj_size if self.proj_size > 0 else self.hidden_size)
        
        layer_len, size = self.D *self.num_layers, self.hidden_size
        self.c_0 = torch.zeros(layer_len, size)
        
        layer_len, size = self.D * self.num_layers, self.output_size
        self.h_0 = torch.zeros(layer_len, size)
        
        if trainable_states:
            self.c_0 = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    self.c_0,
                    a =-(1. / self.c_0.numel()) ** 0.5,
                    b = (1. / self.c_0.numel()) ** 0.5,
                )
            )
            self.h_0 = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    self.h_0,
                    a =-(1. / self.h_0.numel()) ** 0.5,
                    b = (1. / self.h_0.numel()) ** 0.5,
                )
            )
            
        self.seq_len = seq_len
        self.feedback = (
            torch.nn.Identity() if feedback is None and self.input_size == self.output_size else 
            feedback
        )
        
    def forward(self, x, state = None, seq_len = None):
        if len(x.shape) > 2:
            raise RuntimeError(f'Unexpected input size; expected ({self.input_size},) or (N, {self.input_size}); got {x.shape}')
        
        has_batch = len(x.shape) == 2
        if state is None:
            if has_batch:
                h_0 = torch.stack([self.h_0 for _ in x], dim=1)
                c_0 = torch.stack([self.c_0 for _ in x], dim=1)
            else :
                h_0, c_0 = self.h_0, self.c_0
            state = (h_0, c_0)
            
        if seq_len is None:
            seq_len = self.seq_len
            
        if self.return_sequence:
            if len(x.shape) == 1:
                shape = [seq_len, self.output_size]
            else :
                shape = [seq_len, x.shape[0], self.output_size]
                
            seq_track = torch.empty(shape, dtype=x.dtype, device=x.device)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        for _iter in range(seq_len):
            x, state = super().forward(x, state)
            
            if self.return_sequence:
                seq_track[_iter] = x[0]
            
            if _iter != seq_len-1:
                x = self.feedback(x[0]).unsqueeze(0)
            
        if self.return_sequence:
            if not has_batch:
                seq_track = seq_track.squeeze(1)
            elif self._batch_first:
                seq_track = seq_track.swapaxes(0,1)
        else :
            x = x.squeeze(0)
        
        if self.return_sequence and self.return_states:
            return seq_track, state  
        elif self.return_sequence:
            return seq_track
        elif self.return_states:
            return state
        else :
            if has_batch:
                return x
            else :
                return x[0,:]
    
        