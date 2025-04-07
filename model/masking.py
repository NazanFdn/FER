import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class MaskModule(nn.Module):
    """
    Simple placeholder for your actual masking logic:
    a 1x1 conv that outputs a 'mask' of the same shape as x,
    and we do x = x*(1+m) after each layer.
    """
    def __init__(self, in_channels, out_channels, depth=1):
        super(MaskModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depth = depth

    def forward(self, x):
        m = self.conv(x)  # shape: [N, out_channels, H, W]
        m = torch.tanh(m) # activation
        return m

class MaskedResNet(nn.Module):
    """
    Masked ResNet that inserts mask modules after each layer.
    By default uses resnet18 or resnet50 pretrained on ImageNet.
    """
    def __init__(self, arch="resnet18", pretrained=True, num_classes=7, dropout_p=0.3):
        super(MaskedResNet, self).__init__()

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

        # Final classifier
        in_feats = channels[-1]
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_feats, num_classes)
        )

        # Insert mask modules
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


import torch
import torch.nn as nn

from .resnet import BasicBlock, Bottleneck, ResNet
from .utils import load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


from .masking import masking


class ResMasking(ResNet):
    def __init__(self, weight_path):
        super(ResMasking, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000
        )
        # state_dict = torch.load('saved/checkpoints/resnet18_rot30_2019Nov05_17.44')['net']
        # state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
        # self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        """
        # freeze all net
        for m in self.parameters():
            m.requires_grad = False
        """

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56
        m = self.mask1(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer2(x)  # 28
        m = self.mask2(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer3(x)  # 14
        m = self.mask3(x)
        x = x * (1 + m)
        # x = x * m

        x = self.layer4(x)  # 7
        m = self.mask4(x)
        x = x * (1 + m)
        # x = x * m

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


class ResMasking50(ResNet):
    def __init__(self, weight_path):
        super(ResMasking50, self).__init__(
            block=Bottleneck, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000
        )
        # state_dict = torch.load(weight_path)['net']
        state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(2048, 7)

        """
        # freeze all net
        for m in self.parameters():
            m.requires_grad = False
        """

        self.mask1 = masking(256, 256, depth=4)
        self.mask2 = masking(512, 512, depth=3)
        self.mask3 = masking(1024, 1024, depth=2)
        self.mask4 = masking(2048, 2048, depth=1)

    def forward(self, x):  # 224
        x = self.conv1(x)  # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56

        x = self.layer1(x)  # 56
        m = self.mask1(x)
        x = x * (1 + m)

        x = self.layer2(x)  # 28
        m = self.mask2(x)
        x = x * (1 + m)

        x = self.layer3(x)  # 14
        m = self.mask3(x)
        x = x * (1 + m)

        x = self.layer4(x)  # 7
        m = self.mask4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


# def resmasking(in_channels, num_classes, weight_path='saved/checkpoints/resnet18_rot30_2019Nov05_17.44'):
#     return ResMasking(weight_path)


def resmasking(in_channels, num_classes, weight_path=""):
    return ResMasking(weight_path)


def resmasking50_dropout1(in_channels, num_classes, weight_path=""):
    model = ResMasking50(weight_path)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, num_classes))
    return model


def resmasking_dropout1(in_channels=3, num_classes=7, weight_path=""):
    model = ResMasking(weight_path)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
        # nn.Linear(512, num_classes)
    )
    return model


def resmasking_dropout2(in_channels, num_classes, weight_path=""):
    model = ResMasking(weight_path)

    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 7),
    )
    return model


def resmasking_dropout3(in_channels, num_classes, weight_path=""):
    model = ResMasking(weight_path)

    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 128),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(128, 7),
    )
    return model


def resmasking_dropout4(in_channels, num_classes, weight_path=""):
    model = ResMasking(weight_path)

    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(128, 128),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(128, 7),
    )
    return model