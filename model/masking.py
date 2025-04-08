"""
This code is a modified version of ResNet model.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class MaskModule(nn.Module):

    """
    A simple "masking" module intended to learn a mask M for each spatial
    location and channel of the feature maps. Conceptually:
        - Takes in a feature map x (shape [N, C, H, W]).
        - Outputs a mask m of the same shape via a 1×1 convolution.
        - We apply tanh activation to allow the mask values to be in (-1, +1).

    In practice, after each ResNet layer block, we do:
        x = x * (1 + m)
    which can selectively amplify or suppress features based on the mask m.
    """
    def __init__(self, in_channels, out_channels, depth=1):
        """
        Args:
            in_channels (int): Number of input channels to the mask module.
            out_channels (int): Number of output channels (same as in_channels
                                for a direct mask).
            depth (int): Placeholder to potentially extend or stack more layers
                         in the future if needed. Currently unused beyond storing.
        """
        super(MaskModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depth = depth

    def forward(self, x):
        """
        Forward pass to produce and return the mask for x.
        The mask is generated, then passed through tanh to constrain values
        to the range (-1, +1).
        """
        m = self.conv(x)  # shape: [N, out_channels, H, W]
        m = torch.tanh(m) # activation
        return m

class MaskedResNet(nn.Module):
    """
    A modified ResNet (either resnet18 or resnet50) that inserts a 'MaskModule'
    after each of the four main layer blocks. The idea is to scale the feature
    maps by (1 + mask), allowing the network to learn an attention-like or gating
    mechanism for each layer’s output.

    By default, this class loads a pre-trained resnet18 or resnet50 backbone,
    keeps its standard layers, and injects MaskModule objects after each layer.
    Finally, it applies a dropout + linear (fc) head to classify into 'num_classes'.
    """
    def __init__(self, arch="resnet18", pretrained=True, num_classes=7, dropout_p=0.3):
        """
               Args:
                   arch (str): "resnet18" or "resnet50"
                   pretrained (bool): If True, load ImageNet pretrained weights initially.
                   num_classes (int): Number of output classes for the final classifier.
                   dropout_p (float): Dropout probability applied before the final linear layer.
               """
        super(MaskedResNet, self).__init__()
        # Select a backbone and identify the output channel sizes
        if arch == "resnet18":
            backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 128, 256, 512]
        elif arch == "resnet50":
            backbone = resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            channels = [256, 512, 1024, 2048]
        else:
            raise ValueError("Use 'resnet18' or 'resnet50' only.")

        # Keeping standard ResNet layers
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # Creating a final classifier head
        in_feats = channels[-1]
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feats, num_classes)
        )

        # Insert mask modules after each layer
        # Each MaskModule produces a mask for the same shape of the current feature maps.
        self.mask1 = MaskModule(channels[0], channels[0], depth=4)
        self.mask2 = MaskModule(channels[1], channels[1], depth=3)
        self.mask3 = MaskModule(channels[2], channels[2], depth=2)
        self.mask4 = MaskModule(channels[3], channels[3], depth=1)

    def forward(self, x):
        # x expected [N, 3, 48, 48] (already Grayscale->3 channels)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer1
        x = self.layer1(x)
        m = self.mask1(x)
        x = x * (1 + m)

        # layer2
        x = self.layer2(x)
        m = self.mask2(x)
        x = x * (1 + m)

        # layer3
        x = self.layer3(x)
        m = self.mask3(x)
        x = x * (1 + m)

        # layer4
        x = self.layer4(x)
        m = self.mask4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


""" 
####################################
# masked_resnet.py
####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class MaskModule(nn.Module):
    
    Simple placeholder for your actual masking logic:
    a 1x1 conv that outputs a 'mask' of the same shape as x,
    and we do x = x*(1+m) after each layer.
   
    def __init__(self, in_channels, out_channels, depth=1):
        super(MaskModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depth = depth

    def forward(self, x):
        m = self.conv(x)   # shape: [N, out_channels, H, W]
        m = torch.tanh(m)  # Activation
        return m

class MaskedResNet(nn.Module):

    Masked ResNet that inserts mask modules after each layer.
    By default uses resnet18 or resnet50 pretrained on ImageNet.

    def __init__(self, arch="resnet18", pretrained=True, num_classes=7, dropout_p=0.3):
        super(MaskedResNet, self).__init__()

        if arch == "resnet18":
            backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            channels = [64, 128, 256, 512]  # resnet18 channels
        elif arch == "resnet50":
            backbone = resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            channels = [256, 512, 1024, 2048]  # resnet50 channels
        else:
            raise ValueError("Use 'resnet18' or 'resnet50' only.")

        # Keep standard ResNet layers
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool

        # Final classifier: dropout -> linear -> num_classes
        in_feats = channels[-1]
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feats, num_classes)
        )

        # Insert mask modules after each layer
        self.mask1 = MaskModule(channels[0], channels[0], depth=4)
        self.mask2 = MaskModule(channels[1], channels[1], depth=3)
        self.mask3 = MaskModule(channels[2], channels[2], depth=2)
        self.mask4 = MaskModule(channels[3], channels[3], depth=1)

    def forward(self, x):
        # Expect x to be [N,3,48,48] if your dataset is 48x48 grayscale -> 3ch replicate
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        m = self.mask1(x)
        x = x * (1 + m)

        x = self.layer2(x)
        m = self.mask2(x)
        x = x * (1 + m)

        x = self.layer3(x)
        m = self.mask3(x)
        x = x * (1 + m)

        x = self.layer4(x)
        m = self.mask4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
"""
