import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import yaml
import os


from efficientnet import EfficientNet

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
       'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGFeatureExtractor(nn.Module):
    def __init__(self, pretrained):
        super(VGGFeatureExtractor, self).__init__()
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
        if pretrained:
            vgg16_bn.load_state_dict(
                torch.load('./pths/vgg16_bn-6c64b313.pth')
            )
        self.features = vgg16_bn.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained, model_name="efficientnet-b3"):
        super(EfficientNetFeatureExtractor, self).__init__()
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_name(model_name)
        self.features = model

    def forward(self, x):
        features = self.features(x)
        # return features[-4:]
        return features


class VGGFeatureMerger(nn.Module):
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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class FeatureMerger(nn.Module):
    def __init__(self,
                 input_feature_dims=[
                     16, 32, 48, 136, 1536
                 ],
                 inter_out_channels=[
                     128, 64, 32, 32
                 ],
                 out_channels=32):
        """Feature merger module.  It merges the feature maps of different layers from the CNN encoder
        to become a single feature map.

        Args:
            input_feature_dims (list, optional): The channel dimensions of the input feature maps. Defaults to [ 16, 32, 48, 136, 1536 ].
            inter_out_channels (list, optional): The channel dimensions of the intermediate output feature maps. Defaults to [ 128, 64, 32, 32 ].
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
        for i, (in_channels, out_channels) in enumerate(zip(input_feature_dims[:-1][::-1], inter_out_channels)):
            # print(f"i={i}a, in: {prev_channels + in_channels}, out: {out_channels}")
            # print(f"i={i}b, in: {out_channels}, out: {out_channels}")
            setattr(self, f"block_{i+1}_conv_bn_relu_1", ConvBNReLU(
                prev_channels + in_channels, out_channels, 1, padding=0))
            setattr(self, f"block_{i+1}_conv_bn_relu_2", ConvBNReLU(
                out_channels, out_channels, 3, padding=1))
            prev_channels = out_channels

        self.out_conv_bn_relu = ConvBNReLU(prev_channels, out_channels, 3, 1)

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

        y = self.out_conv_bn_relu(y)
        return y


class Output(nn.Module):
    def __init__(self, in_channels=32, scope=512):
        """EAST output module. Takes a feature map from an earlier layer and
        outputs 3 feature maps, one for each output:
            a) score,
            b) location,
            c) angle

        Args:
            input_channels (int, optional): [description]. Defaults to 32.
            scope (int, optional): [description]. Defaults to 512.
        """

        super(Output, self).__init__()

        self.in_channels = in_channels
        self.scope = scope

        # One conv layer, one for each output: a) score, b) location, c) angle
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, 4, 1)
        self.conv3 = nn.Conv2d(in_channels, 1, 1)
        self.scope = scope

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
        angle = (torch.sigmoid(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return score, geo


class EAST(nn.Module):
    def __init__(self,
                 pretrained=True,
                 backbone="efficientnet-b3",
                 scope=512,
                 merger_inter_out_channels=[128, 64, 32, 32],
                 merged_channels=32,
                 ):
        super(EAST, self).__init__()

        self.pretrained = pretrained
        self.backbone = backbone
        self.scope = scope
        self.merger_inter_out_channels = merger_inter_out_channels
        self.merged_channels = merged_channels

        # self.extractor = VGGFeatureExtractor(pretrained)
        self.extractor = EfficientNetFeatureExtractor(pretrained,
                                                      model_name=backbone)

        # Compute extracted feature map channel dimensions
        dummy_out = self.extractor(torch.randn(1, 3, 256, 256))
        feature_dims = [int(t.size(1)) for t in dummy_out]
        print(f"feature_dims: {feature_dims}")

        self.merge = FeatureMerger(input_feature_dims=feature_dims,
                                   inter_out_channels=merger_inter_out_channels,
                                   out_channels=merged_channels)
        self.output = Output(in_channels=merged_channels,
                             scope=scope)

    def forward(self, x):
        x = self.extractor(x)

        # for i, t in enumerate(x):
        #     print(f"extractor output #{i} shape: {t.shape}")

        x = self.merge(x)
        # print(f"merge output shape: {x.shape}")

        x = self.output(x)
        return x

    @staticmethod
    def from_config_file(path="configs/config.yaml"):
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

        # model = EAST()
        return model


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EAST.from_config_file().to(device)
    print(model)

    x = torch.randn(1, 3, 512, 512).to(device)
    score, geo = model(x)

    print(f"score shape   : {score.shape}")
    print(f"geometry shape: {geo.shape}")
