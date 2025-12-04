import torchmetrics
import torch

class OneHotMulticlassAccuracy(torchmetrics.classification.MulticlassAccuracy):
    def __init__(self, num_classes=None, top_k=1, average='macro', multidim_average='global', ignore_index=None, validate_args=True, **kwargs):
        super(OneHotMulticlassAccuracy, self).__init__(num_classes, top_k, average, multidim_average, ignore_index, validate_args, **kwargs)
        
    def forward(self, input, target):
        return super().forward(input, torch.argmax(target,dim=-1))