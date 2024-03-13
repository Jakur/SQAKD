'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]

def printRed(skk): print("\033[91m{}\033[00m" .format(skk))


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super(VGGBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv2d(x)
        if self.batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        self.out = out
        return out


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()

        printRed(f"Create VGG, cfg: {cfg}, batch_norm: {batch_norm}, num_classes: {num_classes}")

        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        
    
    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x):
        h = x.shape[2]
        x = self.block0(x)
        # x = F.relu(x)
        x = self.pool0(x)
        # print(f"block0, before pool: {f0.shape}, after pool: {x.shape}, using pool4: {self.pool4(f0).shape}")
        
        x = self.block1(x)
        # x = F.relu(x)
        x = self.pool1(x)
        # print(f"block1, before pool: {f1.shape}, after pool: {x.shape}, using pool4: {self.pool4(f1).shape}")

        x = self.block2(x)
        # x = F.relu(x)
        x = self.pool2(x)
        # print(f"block2, before pool: {f2.shape}, after pool: {x.shape}, using pool4: {self.pool4(f2).shape}")

        x = self.block3(x)
        # x = F.relu(x)
        # !!!!!!! org, maybe need it when h==64
        # if h == 64:
            # x = self.pool3(x)
        # print(f"block3, before pool: {f3.shape}, after pool: {x.shape}, using pool4: {self.pool4(f3).shape}")

        x = self.block4(x)
        # x = F.relu(x)
        x = self.pool4(x)
        # print(f"block4, before pool: {f4.shape}, after pool: {x.shape}")

        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)

        return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            # if v == 'M':
            #     layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # else:
            #     conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            #     if batch_norm:
            #         layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            #     else:
            #         layers += [conv2d, nn.ReLU(inplace=True)]
            layers.append(VGGBlock(in_channels, v, batch_norm))
            in_channels = v
        # layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]], # vgg-11
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]], # vgg-13
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]], # vgg-16
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]], # vgg-19
    'S': [[64], [128], [256], [512], [512]], # vgg-8
}


def vgg8(pretrained, **kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(pretrained, **kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(pretrained, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


# def vgg11_bn(pretrained, num_classes):
def vgg11_bn(pretrained, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(pretrained, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(pretrained, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(pretrained, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(pretrained, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(pretrained, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(pretrained, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = vgg13_bn(pretrained=False, num_classes=200)
    logit = net(x)

    # for name, p in net.named_parameters():
    #     print(f"{name:50} | {str(p.shape):50} | {p.requires_grad}")

    num_parameters = sum(p.numel() for p in net.parameters())
    # num_trainable_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('\nTotal number of parameters:', num_parameters) # vgg13_bn:9462180