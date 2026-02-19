"""
M3: Automated Testing
- Unit tests for data preprocessing
- Unit tests for model inference utility
Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pytest
from PIL import Image
from torchvision import transforms
from io import BytesIO


# ─────────────────────────────────────────────
# Minimal copy of SimpleCNN for testing
# (avoids importing app.py which loads model file)
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────────
# Preprocessing helper (mirrors train.py / app.py)
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Convert PIL image to model-ready tensor."""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)   # add batch dimension → (1, 3, H, W)


def run_inference(model: nn.Module, tensor: torch.Tensor):
    """Run model inference and return predicted class index and probabilities."""
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()
    return pred, probs


# ═════════════════════════════════════════════
# TEST GROUP 1 — Data Preprocessing
# ═════════════════════════════════════════════

def test_preprocess_rgb_image():
    """RGB image should produce tensor of shape (1, 3, 224, 224)."""
    img    = Image.new("RGB", (300, 200), color=(120, 60, 30))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected shape: {tensor.shape}"


def test_preprocess_grayscale_image():
    """Grayscale (L mode) image must be converted to 3-channel RGB."""
    img    = Image.new("L", (150, 150), color=128)
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), \
        "Grayscale image was not converted to 3-channel tensor"


def test_preprocess_rgba_image():
    """RGBA (4-channel) image must be converted to 3-channel RGB."""
    img    = Image.new("RGBA", (256, 256), color=(10, 20, 30, 255))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), \
        "RGBA image was not converted to 3-channel tensor"


def test_preprocess_output_is_float_tensor():
    """Output tensor must be float32."""
    img    = Image.new("RGB", (100, 100))
    tensor = preprocess_image(img)
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


def test_preprocess_normalization_range():
    """
    After ImageNet normalization, values should be roughly in [-3, 3].
    A plain white image (255,255,255) normalizes to around -2.1 to -2.4.
    """
    img    = Image.new("RGB", (224, 224), color=(255, 255, 255))
    tensor = preprocess_image(img)
    assert tensor.min().item() > -4.0, "Tensor values suspiciously low"
    assert tensor.max().item() <  4.0, "Tensor values suspiciously high"


def test_preprocess_custom_size():
    """Preprocessing should respect a custom image_size."""
    img    = Image.new("RGB", (500, 500))
    tensor = preprocess_image(img, image_size=128)
    assert tensor.shape == (1, 3, 128, 128), \
        f"Expected (1,3,128,128), got {tensor.shape}"


# ═════════════════════════════════════════════
# TEST GROUP 2 — Model Inference
# ═════════════════════════════════════════════

def test_model_output_shape():
    """Model must output logits of shape (1, 2) for a single image."""
    model  = SimpleCNN()
    dummy  = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (1, 2), f"Expected (1,2), got {output.shape}"


def test_inference_returns_valid_class():
    """Predicted class index must be 0 (cat) or 1 (dog)."""
    model        = SimpleCNN()
    img          = Image.new("RGB", (224, 224))
    tensor       = preprocess_image(img)
    pred, probs  = run_inference(model, tensor)
    assert pred in (0, 1), f"Prediction {pred} is not a valid class index"


def test_inference_probabilities_sum_to_one():
    """Softmax probabilities must sum to 1.0."""
    model       = SimpleCNN()
    img         = Image.new("RGB", (224, 224))
    tensor      = preprocess_image(img)
    _, probs    = run_inference(model, tensor)
    total       = probs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_inference_probabilities_in_range():
    """Each probability must be between 0 and 1."""
    model      = SimpleCNN()
    img        = Image.new("RGB", (224, 224))
    tensor     = preprocess_image(img)
    _, probs   = run_inference(model, tensor)
    for i, p in enumerate(probs):
        assert 0.0 <= p.item() <= 1.0, \
            f"Probability at index {i} is {p.item()}, out of [0,1]"


def test_model_batch_inference():
    """Model must handle a batch of 4 images."""
    model  = SimpleCNN()
    batch  = torch.randn(4, 3, 224, 224)
    output = model(batch)
    assert output.shape == (4, 2), f"Expected (4,2), got {output.shape}"


def test_model_eval_mode_no_grad():
    """Inference in eval mode must not raise errors."""
    model  = SimpleCNN()
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    try:
        with torch.no_grad():
            output = model(tensor)
    except Exception as e:
        pytest.fail(f"Inference in eval mode raised an exception: {e}")
