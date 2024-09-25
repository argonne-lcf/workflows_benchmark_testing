import torch
import torch.nn as nn

def count_weights(model: nn.Module) -> int:
    """Count number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



