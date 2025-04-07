import pytest
import torch
from model.masking import MaskedResNet

@pytest.fixture
def model():
    # Load the model as a fixture
    model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=7, dropout_p=0.3)
    return model

def test_model_output_shape(model):
    # Generate a random input tensor of shape [1, 3, 48, 48] (batch size 1, 3 channels, 48x48)
    input_tensor = torch.randn(1, 3, 48, 48)
    output = model(input_tensor)
    # Check the output shape (should be [1, 7] for 7 classes)
    assert output.shape == (1, 7), f"Output shape mismatch: {output.shape}"
