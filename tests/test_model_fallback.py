import sys
from unittest.mock import MagicMock, patch

# Mock torch and torchvision before importing clip
mock_torch = MagicMock()
mock_torch.__version__ = "2.0.0"
mock_torchvision = MagicMock()
mock_pil = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_torch.nn.functional
sys.modules["torchvision"] = mock_torchvision
sys.modules["torchvision.transforms"] = mock_torchvision.transforms
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

import clip


def test_load_fallback_on_attribute_error(tmp_path):
    dummy_file = tmp_path / "dummy.pt"
    dummy_file.write_bytes(b"dummy_content")

    # When torch.jit.load raises AttributeError, clip.load should catch it, seek to 0, and load via torch.load
    mock_torch.jit.load.side_effect = AttributeError("type object 'RecursiveScriptModule' has no attribute '_construct'")
    mock_model = MagicMock()
    mock_torch.load.return_value = {"state": 1}

    with patch("clip.clip.build_model", return_value=mock_model) as mock_build:
        model, transform = clip.load(str(dummy_file), device="cpu", jit=False)
        assert model is mock_model.to.return_value
        mock_torch.load.assert_called_once()
        mock_build.assert_called_once_with({"state": 1})
