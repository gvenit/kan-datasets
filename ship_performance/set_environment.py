# import torch

# torch.backends.fp32_precision = "ieee"

# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.fp32_precision = "ieee"
#     torch.backends.cuda.matmul.allow_tf32 = False
    
# if torch.backends.cudnn.is_available():
#     torch.backends.cudnn.fp32_precision = "ieee"
#     torch.backends.cudnn.conv.fp32_precision = "ieee"
#     torch.backends.cudnn.rnn.fp32_precision = "ieee"
    
#    torch.backends.cudnn.allow_tf32 = False

# torch.set_float32_matmul_precision('high')