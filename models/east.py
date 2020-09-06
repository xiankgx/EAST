import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .deeplapv3plus import (deeplabv3plus_mobilenet, deeplabv3plus_resnet50,
                            deeplabv3plus_resnet101)
from .efficientnet import EfficientNetFeat
from .mobilenet import InvertedResidual, mobilenet_v2
from .pvanet import PVANetFeat
from .res2net_v1b import (res2net50_v1b_feature_extractor,
                          res2net101_v1b_feature_extractor)
from .resnet import resnet34, resnet50, resnet101
from .u2net import U2NET, U2NETP
from .vgg import vgg16_bn
from .xception import pretrained_settings as xception_pretrained_settings
from .xception import xception_feature_extractor

preprocessing_params = {
    # VGG
    "vgg16_bn": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5)
    },

    "resnet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "res2net": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "xception": {
        "mean": xception_pretrained_settings["xception"]["imagenet"]["mean"],
        "std": xception_pretrained_settings["xception"]["imagenet"]["std"]
    },

    "efficientnet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "pvanet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "mobilenetv2": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "u2net": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },

    "deeplabv3plus": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    },
}

###############################################################################
# Common
###############################################################################


class ConvBNReLU(nn.Module):
    """A commonly used block in CNNs consisting of a 2D-convolution layer, followed by batch normaliztion and ReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

###############################################################################
# Backbone
###############################################################################


class VGG16BNFeat(nn.Module):
    """VGG16 BN feature extractor for EAST"""

    def __init__(self, pretrained):
        super(VGG16BNFeat, self).__init__()
        backbone = vgg16_bn(pretrained)
        self.features = backbone.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)

        # In our experiments, we also adopted the
        # well-known VGG16 [32] model, where feature maps after
        # pooling-2 to pooling-5 are extracted.
        return out[1:]


class VGGFeatureMerger(nn.Module):
    """VGG feature merger for EAST"""

    def __init__(self):
        super(VGGFeatureMerger, self).__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y


class PVANetFeat4EAST(PVANetFeat):
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)  # 1/4 feature
        x2 = self.conv3(x1)  # 1/8
        x3 = self.conv4(x2)  # 1/16
        x4 = self.conv5(x3)  # 1/32
        return [x1, x2, x3, x4]


class MobileNetV2Feat(nn.Module):
    """MobileNet v2 feature extractor for EAST"""

    def __init__(self, pretrained=True):
        super(MobileNetV2Feat, self).__init__()
        backbone = mobilenet_v2(pretrained)
        self.features = backbone.features
        self.extract_layers = [3, 6, 13, 18]

    def forward(self, x):
        out = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.extract_layers:
                out.append(x)
            # print(f">>> layer i: {i}, output shape: {x.shape}")

        return out


def vgg16_bn_feature_extractor(pretrained: bool):
    """Create a VGG16 BN feature extractor model for EAST."""
    model = VGG16BNFeat(pretrained)
    return model


def efficientnet_feature_extractor(pretrained: bool = True, model_name: str = "efficientnet-b3"):
    """Create a EfficientNet feature extractor model for EAST."""
    if pretrained:
        model = EfficientNetFeat.from_pretrained(model_name)
    else:
        model = EfficientNetFeat.from_name(model_name)
    return model


def pvanet_feature_extractor(pretrained: bool):
    """Create a PVANet feature extractor model for EAST."""
    model = PVANetFeat4EAST()
    if pretrained:
        print("Warning, no pretrained weights for PVANet backbone! Note: This is not an error.")
    return model


def mobilenetv2_feature_extractor(pretrained: bool):
    model = MobileNetV2Feat(pretrained)
    return model


def u2net_feature_extractor(pretrained: bool, portable=False):
    if portable:
        model = U2NETP()

        if pretrained:
            print("Warning, no pretrained weights for portable U2Net backbone.")
    else:
        model = U2NET()

        if pretrained:
            model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          "u2net.pth"),
                                             map_location="cpu"),
                                  strict=False)

    return model


###############################################################################
# EAST model and sub-modules
###############################################################################


class FeatureMerger(nn.Module):
    def __init__(self,
                 input_feature_dims,
                 inter_out_channels=[128, 64, 32],
                 out_channels: int = 32):
        """Feature merger module. It merges the feature maps of different layers from the CNN encoder
        to become a single feature map.

        Args:
            input_feature_dims (List[int], optional): The channel dimensions of the input feature maps. Defaults to [ 16, 32, 48, 136, 1536 ].
            inter_out_channels (List[int], optional): The channel dimensions of the intermediate output feature maps. Defaults to [ 128, 64, 32, 32 ].
            out_channels (int, optional): The channel dimension of the output feature map. Defaults to 32.
        """

        super(FeatureMerger, self).__init__()

        self.input_feature_dims = input_feature_dims
        print(f"input_feature_dims: {input_feature_dims}")

        if len(inter_out_channels) + 1 != len(input_feature_dims):
            raise ValueError(
                "Length of merged channels must be 1 less than length of input_feature_dims.")

        self.input_feature_dims = input_feature_dims
        self.inter_out_channels = inter_out_channels
        self.out_channels = out_channels

        prev_channels = input_feature_dims[-1]
        for i, (cin, cout) in enumerate(zip(input_feature_dims[:-1][::-1], inter_out_channels)):
            print(f"prev_channels: {prev_channels}, cin: {cin}, cout: {cout}")
            print(f"i={i}a, in: {prev_channels + cin}, out: {cout}")
            print(f"i={i}b, in: {cout}, out: {cout}")
            setattr(self, f"block_{i+1}_conv_bn_relu_1",
                    ConvBNReLU(prev_channels + cin, cout, 1, padding=0))
            setattr(self, f"block_{i+1}_conv_bn_relu_2",
                    ConvBNReLU(cout, cout, 3, padding=1))
            prev_channels = cout

        self.out_conv_bn_relu = ConvBNReLU(
            prev_channels, out_channels, 3, padding=1)

    def forward(self, x):
        y = x[-1]
        for i in range(len(self.input_feature_dims[:-1])):
            y = F.interpolate(y, scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
            # print(f"y.shape: {y.shape}")

            layer1 = getattr(self, f"block_{i+1}_conv_bn_relu_1")
            layer2 = getattr(self, f"block_{i+1}_conv_bn_relu_2")
            # print(f">>> k-i: {k - i}")
            # print(x[k - i].shape)
            y = torch.cat((y, x[len(self.input_feature_dims) - 2 - i]), 1)
            y = layer1(y)
            y = layer2(y)
            # print(f"y shape: {y.shape}")

        y = self.out_conv_bn_relu(y)
        return y


class Output(nn.Module):
    def __init__(self,
                 in_channels: int = 32,
                 scope: int = 512):
        """EAST output module. Takes a feature map from an earlier layer and
        outputs 3 feature maps, one for each output:
            a) score,
            b) location,
            c) angle

        Args:
            input_channels (int, optional): Number of input channels. Defaults to 32.
            scope (int, optional): Model input size (width/height). Defaults to 512.
        """

        super(Output, self).__init__()

        self.in_channels = in_channels
        self.scope = scope

        # One conv layer, one for each output: a) score, b) location, c) angle
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, 4, 1)
        self.conv3 = nn.Conv2d(in_channels, 1, 1)

        # Weights intialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = torch.sigmoid(self.conv1(x))

        loc = torch.sigmoid(self.conv2(x)) * self.scope

        # 2pi - 360 deg
        # 0.5pi -> x = (0.5 * 360)/2 = 90 deg
        # angle can only be in the range [-90, 90] degrees
        angle = (torch.sigmoid(self.conv3(x)) - 0.5) * math.pi

        # Concat loc and angle map as a single geo map
        geo = torch.cat((loc, angle), 1)

        return score, geo


class EAST(nn.Module):
    """Zhou, X., Yao, C., Wen, H., Wang, Y., Zhou, S., He, W., & Liang, J. (2017).
    EAST: An Efficient and Accurate Scene Text Detector.
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017-Janua, 2642–2651.
    https://doi.org/10.1109/CVPR.2017.283
    """

    def __init__(self,
                 pretrained: bool = True,
                 backbone: str = "efficientnet-b3",
                 scope: int = 512,
                 merger_inter_out_channels=[128, 64, 32],
                 merged_channels: int = 32,
                 ):
        super(EAST, self).__init__()

        self.pretrained = pretrained
        self.backbone = backbone
        self.scope = scope
        self.merger_inter_out_channels = merger_inter_out_channels
        self.merged_channels = merged_channels
        self.model_type = "east"

        # Create backbone model
        if "deeplabv3plus" in backbone:
            backbone = backbone.split("_")[-1]

            deeplabv3plus_params = {
                "num_classes": merged_channels,
                "pretrained_backbone": pretrained
            }

            if backbone == "resnet50":
                self.extractor = deeplabv3plus_resnet50(
                    **deeplabv3plus_params)
            elif backbone == "resnet101":
                self.extractor = deeplabv3plus_resnet101(
                    **deeplabv3plus_params)
            elif backbone == "mobilenetv2":
                self.extractor = deeplabv3plus_mobilenet(
                    **deeplabv3plus_params)
            else:
                raise ValueError(
                    f"Unknown backbone for DeepLabV3Plus: {backbone}")

            self.model_type = "deeplabv3plus"

        elif backbone == "vgg16_bn":
            self.extractor = vgg16_bn_feature_extractor(pretrained)

        elif backbone in [f"efficientnet-b{i}" for i in range(8)]:
            self.extractor = efficientnet_feature_extractor(pretrained,
                                                            model_name=backbone)

        elif backbone == "pvanet":
            self.extractor = pvanet_feature_extractor(pretrained)

        elif backbone == "xception":
            self.extractor = xception_feature_extractor(pretrained)

        elif "resnet" in backbone:
            if backbone == "resnet34":
                self.extractor = resnet34(pretrained)
            elif backbone == "resnet50":
                self.extractor = resnet50(pretrained)
            elif backbone == "resnet101":
                self.extractor = resnet101(pretrained)
            else:
                raise ValueError(f"Unknown resnet backbone: {backbone}")

        elif "res2net" in backbone:
            if backbone == "res2net50_v1b":
                self.extractor = res2net50_v1b_feature_extractor(pretrained)
            elif backbone == "res2net101_v1b":
                self.extractor = res2net101_v1b_feature_extractor(pretrained)
            else:
                raise ValueError(f"Unknown res2net backbone: {backbone}")

        elif "mobilenetv2" in backbone:
            self.extractor = mobilenetv2_feature_extractor(pretrained)

        elif "u2net" in backbone:
            if backbone == "u2net":
                self.extractor = u2net_feature_extractor(pretrained, False)
            elif backbone == "u2netp":
                self.extractor = u2net_feature_extractor(pretrained, True)
            else:
                raise ValueError(f"Unknown u2net backbone: {backbone}")

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if self.model_type == "east":
            # Compute extracted feature map channel dimensions
            dummy_out = self.extractor(torch.randn(1, 3, 256, 256))
            feature_dims = [int(t.size(1)) for t in dummy_out]
            print(f"feature_dims: {feature_dims}")

            # Create feature merger and output modules
            self.merge = FeatureMerger(input_feature_dims=feature_dims,
                                       inter_out_channels=merger_inter_out_channels,
                                       out_channels=merged_channels)
        self.output = Output(in_channels=merged_channels,
                             scope=scope)

    def forward(self, x):
        # EAST model with encoder-decoder
        if self.model_type == "east":
            x = self.extractor(x)
            x = self.merge(x)

        # DeepLabV3Plus model with encoder-decoder and atrous convolution
        elif self.model_type == "deeplabv3plus":
            x = self.extractor(x)

        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")

        x = self.output(x)
        return x

    @staticmethod
    def from_config_file(path: str = "../configs/config.yaml"):
        """Instantiate model from config file.

        Args:
            path (str, optional): Config file. Defaults to "configs/config.yaml".

        Returns:
            nn.Module: An instance of the EAST model.
        """

        if not os.path.isfile(path):
            print(f"Config file not found: {path}")

        with open(path, "r") as f:
            conf = yaml.load(f, Loader=yaml.Loader)
        model_conf = conf["model"]

        model = EAST(**model_conf)
        return model

    def get_preprocessing_params(self):
        """Get input preprocessing (input normalization) parameters that are used to train the model."""

        if "efficientnet" in self.backbone:
            return preprocessing_params["efficientnet"]
        elif "deeplabv3plus" in self.backbone:
            return preprocessing_params["deeplabv3plus"]
        elif "resnet" in self.backbone:
            return preprocessing_params["resnet"]
        elif "res2net" in self.backbone:
            return preprocessing_params["res2net"]
        elif "u2net" in self.backbone:
            return preprocessing_params["u2net"]
        else:
            return preprocessing_params[self.backbone]


###############################################################################

if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # model = EAST(True, "deeplabv3plus_resnet50", 32)
    # model = EAST.from_config_file().to(device)
    # model = MobileNetV2Feat()
    model = EAST(backbone="u2net")
    print(model)

    x = torch.randn(2, 3, 512, 512).to(device)
    # out = model(x)
    # for t in out:
    #     print(t.shape)
    score, geo = model(x)

    scale = score.size(2)/512

    print(f"score shape   : {score.shape}")
    print(f"geometry shape: {geo.shape}")
    print(f"scale         : {scale}")
