import importlib.util

import torch


def get_device_initial(preferred_device=None):
    """
    Determine the appropriate device to use (cuda, hpu, or cpu).
    Args:
        preferred_device (str): User-preferred device ('cuda', 'hpu', or 'cpu').

    Returns:
        str: Device string ('cuda', 'hpu', or 'cpu').
    """
    # Check for HPU support
    if importlib.util.find_spec("habana_frameworks") is not None:
        from habana_frameworks.torch.utils.library_loader import load_habana_module

        load_habana_module()
        if torch.hpu.is_available():
            if preferred_device == "hpu" or preferred_device is None:
                return "hpu"

    # Check for CUDA (GPU support)
    if torch.cuda.is_available():
        if preferred_device == "cuda" or preferred_device is None:
            return "cuda"

    # Default to CPU
    return "cpu"
