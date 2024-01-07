from clip.clip import tokenize as _tokenize, load as _load, available_models as _available_models
import re
import string

dependencies = ["torch", "torchvision", "ftfy", "regex", "tqdm"]

# For compatibility (cannot include special characters in function name)
model_functions = { model: re.sub(f'[{string.punctuation}]', '_', model) for model in _available_models()}

def _create_hub_entrypoint(model):
    def entrypoint(**kwargs):      
        return _load(model, **kwargs)
    
    entrypoint.__doc__ = f"""Loads the {model} CLIP model

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to put the loaded model

        jit : bool
            Whether to load the optimized JIT model or more hackable non-JIT model (default).

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"

        Returns
        -------
        model : torch.nn.Module
            The {model} CLIP model

        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        """
    return entrypoint

def tokenize():
    return _tokenize

_entrypoints = {model_functions[model]: _create_hub_entrypoint(model) for model in _available_models()}

globals().update(_entrypoints)