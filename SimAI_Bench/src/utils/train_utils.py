import torch
import torch.nn as nn

def count_weights(model: nn.Module) -> int:
    """Count number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_cuda_mem_usage(device: torch.device) -> float:
    """Get the memory usage on a CUDA device in MB
    """
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024 ** 2

