import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

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
        return features[-4:]


class FeatureMerger(nn.Module):
    def __init__(self):
        super(FeatureMerger, self).__init__()

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


class EfficientNetFeatureMerger(nn.Module):
    def __init__(self,
                 feature_dims=[
                     32, 48, 136, 1536
                 ],
                 merged_channels=[
                     128, 64, 32
                 ]):
        super(EfficientNetFeatureMerger, self).__init__()

        self.conv1 = nn.Conv2d(
            feature_dims[-1] + feature_dims[-2], merged_channels[0], 1)
        self.bn1 = nn.BatchNorm2d(merged_channels[0])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            merged_channels[0], merged_channels[0], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(merged_channels[0])
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            merged_channels[0] + feature_dims[-3], merged_channels[1], 1)
        self.bn3 = nn.BatchNorm2d(merged_channels[1])
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            merged_channels[1], merged_channels[1], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(merged_channels[1])
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(
            merged_channels[1] + feature_dims[-4], merged_channels[2], 1)
        self.bn5 = nn.BatchNorm2d(merged_channels[2])
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            merged_channels[2], merged_channels[2], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(merged_channels[2])
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(
            merged_channels[2], merged_channels[2], 3, padding=1)
        self.bn7 = nn.BatchNorm2d(merged_channels[2])
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


class Output(nn.Module):
    def __init__(self, scope=512):
        super(Output, self).__init__()

        self.conv1 = nn.Conv2d(32, 1, 1)
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.scope = 512

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
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        # self.extractor = VGGFeatureExtractor(pretrained)
        self.extractor = EfficientNetFeatureExtractor(pretrained,
                                                      model_name="efficientnet-b3")

        # Compute extracted feature map channel dimensions
        dummy_out = self.extractor(torch.randn(1, 3, 256, 256))
        feature_dims = [int(t.size(1)) for t in dummy_out]

        self.merge = EfficientNetFeatureMerger(feature_dims=feature_dims)
        self.output = Output()

    def forward(self, x):
        x = self.extractor(x)

        # for i, t in enumerate(x):
        #     print(f"extractor output #{i} shape: {t.shape}")

        x = self.merge(x)
        # print(f"merge output shape: {x.shape}")

        x = self.output(x)
        return x


if __name__ == '__main__':
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)
