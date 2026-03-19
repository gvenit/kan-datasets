import torch
from typing import Literal

# @torch.compile()
class RNNFFT (torch.nn.Module):
    def __init__(self, input_size, radix = 2, residual : int = True, mode : Literal['DIT','DIF']= 'DIT'):
        super(RNNFFT,self).__init__()
        
        self.input_size = input_size
        self.radix = radix
        self.weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(self.radix, self.input_size//self.radix,)
        ))
        if self.input_size > self.radix:
            self.recursive = RNNFFT(self.input_size//self.radix, radix=self.radix, residual=int(residual)-1) 
        else : 
            self.recursive = torch.nn.Identity()
            
        self.lin_weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(radix, radix)
        ))
        self.residual = bool(residual)
        self.drop = torch.nn.Dropout(0.5)
        
        if mode == 'DIF':
            self.forward = self.__DIF
        elif mode == 'DIT':
            self.forward = self.__DIT
        else :
            raise ValueError(f'Expected mode to be one of "DIT", "DIF"; got {mode}')
        
    # Decimation in frequency
    def __DIF(self, x):
        assert x.shape[-1] == self.input_size
        
        if self.residual :
            old = x

        x_shape = x.shape
        x = self.recursive(
            torch.einsum(
                'ij,...ik->...jk',
                self.lin_weights,
                torch.einsum(
                    'ij,...ij->...ij',
                    self.weights,
                    self.drop(x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,)))
                ),
            )
        ).reshape(x_shape)
        if self.residual :
            x = old + x
        return x
    
    # Decimation in time
    def __DIT(self, x):
        assert x.shape[-1] == self.input_size
        
        if self.residual :
            old = x

        x_shape = x.shape
        # x = torch.einsum(
        #     'ij,...ik->...jk',
        #     self.lin_weights,
        #     torch.einsum(
        #         'ij,...ij->...ij',
        #         self.weights,
        #         self.recursive(
        #             self.drop(x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,)))
        #         )
        #     ),
        # ).reshape(x_shape)
        x = torch.einsum(
            'ij,...ik->...jk',
            self.lin_weights,
            self.weights * self.recursive(
                    self.drop(x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,)))
            ),
        ).reshape(x_shape)
        
        if self.residual :
            return old + x
        return x
    
    # def forward(self, x):
    #     assert x.shape[-1] == self.input_size
    #     x_shape = x.shape
    #     x = self.recursive(
    #         x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,))
    #     )
    #     x = torch.einsum(
    #         'ij,...ij->...ij',
    #         self.weights,
    #         x
    #     )
    #     x = torch.einsum(
    #         'ij,...ik->...jk',
    #         self.lin_weights,
    #         x,
    #     )
    #     return x.reshape(x_shape)
        
class RNNFFT2 (torch.nn.Module):
    def __init__(self, input_size, radix = 2, residual : int = True, mode : Literal['DIT','DIF']= 'DIF'):
        super(RNNFFT2,self).__init__()
        
        self.input_size = input_size
        self.radix = radix
        self.weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(self.radix, self.input_size//self.radix,)
        ))
        if self.input_size > self.radix:
            self.recursive = torch.nn.ModuleList([
                RNNFFT2(self.input_size//self.radix, radix=self.radix, residual=int(residual)-1) 
                    for _ in range(self.radix)
            ])
        else : 
            self.recursive = torch.nn.ModuleList([
                torch.nn.Identity()
                    for _ in range(self.radix)
            ])
            
        self.lin_weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(radix, radix)
        ))
        self.residual = bool(residual)
        self.drop = torch.nn.Dropout(0.5)
        
        if mode == 'DIF':
            self.forward = self.__DIF
        elif mode == 'DIT':
            self.forward = self.__DIT
        else :
            raise ValueError(f'Expected mode to be one of "DIT", "DIF"; got {mode}')
        
    # Decimation in frequency
    def __DIF(self, x):
        assert x.shape[-1] == self.input_size
        
        if self.residual :
            old = x

        x_shape = x.shape
        x = self.recursive(
            torch.einsum(
                'ij,...ik->...jk',
                self.lin_weights,
                torch.einsum(
                    'ij,...ij->...ij',
                    self.weights,
                    
                    self.drop(x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,)))
                ),
            )
        ).reshape(x_shape)
        if self.residual :
            x = old + x
        return x
    
    # Decimation in time
    def __DIT(self, x):
        assert x.shape[-1] == self.input_size
        
        if self.residual :
            old = x

        x_shape = x.shape
        x = torch.einsum(
            'ij,...ik->...jk',
            self.lin_weights,
            torch.einsum(
                'ij,...ij->...ij',
                self.weights,
                self.recursive(
                    self.drop(x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,)))
                )
            ),
        ).reshape(x_shape)
        
        if self.residual :
            return old + x
        return x
    
    # def forward(self, x):
    #     assert x.shape[-1] == self.input_size
    #     x_shape = x.shape
    #     x = self.recursive(
    #         x.reshape(x.shape[:-1] + (self.radix, self.input_size//self.radix,))
    #     )
    #     x = torch.einsum(
    #         'ij,...ij->...ij',
    #         self.weights,
    #         x
    #     )
    #     x = torch.einsum(
    #         'ij,...ik->...jk',
    #         self.lin_weights,
    #         x,
    #     )
    #     return x.reshape(x_shape)
        
class NNFFTLayer (torch.nn.Module):
    def __init__(self, input_size, radix = 2, dilation = 1):
        super(NNFFTLayer,self).__init__()
        
        self.weights = torch.nn.Parameter(
            torch.ones((2*dilation)) / input_size ** 0.5
        )
        self.input_size = input_size
        self.radix = radix
        self.dilation = dilation
        self.lin_weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(radix, radix)
        ))
        
    def forward(self, x):
        assert x.shape[-1] == self.input_size
        
        x = self.weights.expand(self.radix, self.weights.size(0)).reshape(-1) * x
        
        x = torch.split(x, int(self.input_size/self.dilation), dim=-1)
        
        x = (
            torch.cat(x[offset::self.radix])
                for offset in range(self.radix)
        )
        x = torch.cat([
                torch.sum(
                        torch.stack([
                        w[i] * x_i
                            for i, x_i in enumerate(x)
                        ], dim = 0
                    ), dim=0
                ) for w in self.lin_weights
            ], dim=-1
        )
        x = torch.split(x, int(self.input_size/self.dilation), dim=-1)
        
        return torch.cat((
            torch.cat(x[offset::self.input_size/self.radix])
                for offset in range(self.input_size/self.radix)
        ))
        
class NNFFT (torch.nn.Module):
    def __init__(self, input_size, radix = 2):
        super(NNFFT,self).__init__()
        
        self.modules = torch.nn.ModuleList(
            NNFFTLayer(
                input_size=input_size,
                radix=radix,
                dilation=radix ** i
            )
            for i in range(int(torch.log(input_size) / torch.log(radix)))
        )
        
        self.act = torch.nn.Identity()
        
    def forward(self,x):
        for op in self.modules:
            x = self.act(op(x))
            
        return x
     
class RecurrentNNFFT (torch.nn.Module):
    def __init__(self, input_size, radix = 2):
        super(RecurrentNNFFT,self).__init__()
        
        self.modules = torch.nn.ModuleList(
            NNFFTLayer(
                input_size=input_size,
                radix=radix,
                dilation=radix ** i
            )
            for i in range(int(torch.log(input_size) / torch.log(radix)))
        )
        
        self.act = torch.nn.Identity()
        
    def forward(self,x):
        for op in self.modules:
            x = x + self.act(op(x))
            
        return x
    
class OptimisedRNNFFT(torch.nn.Module):
    def __init__(self, input_size, radix=2, residual : int = True):
        super(OptimisedRNNFFT, self).__init__()
        
        self.weights = torch.nn.Parameter(
            torch.ones(input_size) / input_size ** 0.5
        )
        
        self.input_size = input_size
        self.radix = radix
        
        if input_size > radix:
            self.recursive = OptimisedRNNFFT(input_size // radix, radix=radix, residual=int(residual)-1)
        else:
            self.recursive = torch.nn.Identity()
        
        self.lin_weights = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.ones(radix, radix)
            )
        )
        self.residual = bool(residual)
        
    def forward(self, x):
        assert x.shape[-1] == self.input_size, \
            f"Input size {x.shape[-1]} doesn't match expected {self.input_size}"
        
        chunk_size = self.input_size // self.radix
        
        x_reshaped = x.reshape(*x.shape[:-1], self.radix, chunk_size)
       
        batch_dims = x_reshaped.shape[:-2]  # Preserve batch dimensions
        total_chunks = torch.prod(torch.tensor(batch_dims)) * self.radix if batch_dims else self.radix
        
        x_flat = x_reshaped.reshape(total_chunks, chunk_size)
        x_recurrent_flat = self.recursive(x_flat)
        x_recurrent = x_recurrent_flat.reshape(*batch_dims, self.radix, chunk_size)
        
        x_weighted = self.weights * x_recurrent.reshape(*x_recurrent.shape[:-2], -1)
        
        x_butterfly = x_weighted.reshape(*x_weighted.shape[:-1], self.radix, chunk_size)
        
        result = torch.einsum('ij,...jc->...ic', self.lin_weights, x_butterfly)
        result.reshape(*result.shape[:-2], -1)
        
        if self.residual :
            result += x

        return result
    