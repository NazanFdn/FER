
import unittest
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as T
from model.masking import MaskedResNet, MaskModule


# If the preprocess_image function is defined in the same module or another module,
# import it appropriately. Here, we assume it is defined in the current scope.
def preprocess_image(face_roi):
    """
    Preprocess the detected face ROI ResNet model:
      1) Convert to grayscale.
      2) Resize to 48x48.
      3) Normalize pixel values to [0,1].
      4) Replicate to 3 channels (ResNet expects 3 channels).
      5) Convert to a torch.Tensor and normalize with ImageNet stats.
      6) Add a batch dimension so that shape becomes (1, 3, 48, 48).
    """
    # Convert from BGR to grayscale
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    face_resized = cv2.resize(face_gray, (48, 48))
    # Normalize pixel values to [0,1]
    face_normalized = face_resized / 255.0
    # Expand grayscale image to 3 channels
    face_normalized = np.repeat(face_normalized[:, :, np.newaxis], 3, axis=2)
    # Convert to a PyTorch tensor
    face_tensor = T.ToTensor()(face_normalized).type(torch.float32)
    # Normalize using ImageNet mean and std
    face_tensor = T.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])(face_tensor)
    # Add batch dimension: (1, 3, 48, 48)
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor


class TestMaskModule(unittest.TestCase):
    def test_mask_module_output_shape(self):
        """
        Test that the MaskModule produces an output mask with the same dimensions as its input.
        """
        batch_size = 2
        channels = 32
        height, width = 12, 12
        dummy_input = torch.randn(batch_size, channels, height, width)
        # Create a MaskModule instance that expects 32 input and output channels.
        mask_module = MaskModule(in_channels=channels, out_channels=channels, depth=1)
        # Forward the dummy input through the mask module.
        mask = mask_module(dummy_input)
        # Verify the output mask has the same shape as the input.
        self.assertEqual(mask.shape, dummy_input.shape,
                         "Output mask shape must equal the input shape.")

    def test_mask_module_value_range(self):
        """
        Test that the values output by MaskModule are in the range [-1, +1] (due to tanh activation).
        """
        batch_size = 2
        channels = 16
        height, width = 10, 10
        dummy_input = torch.randn(batch_size, channels, height, width)
        mask_module = MaskModule(in_channels=channels, out_channels=channels, depth=1)
        mask = mask_module(dummy_input)
        # Assert that all values in the mask are at most 1 and at least -1.
        self.assertTrue(torch.all(mask <= 1), "All mask values should be <= 1")
        self.assertTrue(torch.all(mask >= -1), "All mask values should be >= -1")


class TestMaskedResNet(unittest.TestCase):
    def test_masked_resnet_forward_shape(self):
        """
        Test the forward pass of MaskedResNet.
        Create a dummy input with shape [batch_size, 3, 48, 48] and verify that
        the output has shape [batch_size, num_classes].
        """
        batch_size = 4
        num_classes = 7
        dummy_input = torch.randn(batch_size, 3, 48, 48)
        # Instantiate MaskedResNet with ResNet-18 backbone and 7 output classes.
        model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=num_classes, dropout_p=0.3)
        outputs = model(dummy_input)
        # Check that output shape is (batch_size, num_classes)
        self.assertEqual(outputs.shape, (batch_size, num_classes),
                         "Output shape should be [batch_size, num_classes]")

    def test_masked_resnet_backward(self):
        """
        Test that gradients can be computed through MaskedResNet.
        This ensures that all parts of the network, including the masking modules,
        are connected properly for backpropagation.
        """
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 48, 48)
        model = MaskedResNet(arch="resnet18", pretrained=False, num_classes=7, dropout_p=0.3)
        outputs = model(dummy_input)
        targets = torch.tensor([0, 1])  # Dummy target class indices
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        loss.backward()
        grad_found = any(param.grad is not None for param in model.parameters())
        self.assertTrue(grad_found, "Gradients should be computed for model parameters.")


class TestPreprocessImage(unittest.TestCase):
    def test_preprocess_image_output(self):
        """
        Test that preprocess_image converts an input ROI (NumPy array) to a torch.Tensor
        with shape (1, 3, 48, 48) and type torch.float32.
        """
        import cv2
        # Create a dummy colored image (simulate a face ROI), 60x60 in BGR format.
        dummy_roi = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
        processed_tensor = preprocess_image(dummy_roi)
        self.assertEqual(processed_tensor.shape, (1, 3, 48, 48),
                         "Processed tensor should have shape (1, 3, 48, 48).")
        self.assertEqual(processed_tensor.dtype, torch.float32,
                         "Processed tensor should be of type torch.float32.")


if __name__ == "__main__":
    unittest.main()
