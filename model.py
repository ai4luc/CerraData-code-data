import torch
import math
from torch import nn
from torch.nn import functional as F
import torchvision

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    __constants__ = ['model_name', 'num_classes', 'num_domains']

    def __init__(self, backbone, num_classes):
        super(Model, self).__init__()

        if not isinstance(backbone, nn.Module):
             raise ValueError('A model must be provided.')

        self.num_classes = num_classes
        self.model_name = backbone.__class__.__name__.lower()

        if any(prefix in self.model_name for prefix in ['alexnet', 'mnasnet', 'mobilenet', 'vgg', 'convnext', 'efficient']):
            feature_dim = backbone.classifier[-1].in_features
            layer = nn.Linear(feature_dim, num_classes)

            if 'efficient' in self.model_name:
                init_range = 1.0 / math.sqrt(layer.out_features)
                nn.init.uniform_(layer.weight, -init_range, init_range)
                nn.init.zeros_(layer.bias)

            if 'convnext' in self.model_name:
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            backbone.classifier[-1] = layer
        elif 'densenet' in self.model_name:
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Linear(feature_dim, num_classes)
        elif any(prefix in self.model_name for prefix in ['googlenet', 'inception', 'resnet', 'shufflenet', 'resnext']):
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Linear(feature_dim, num_classes)
        elif 'squeezenet' in self.model_name:
            in_channels = backbone.classifier[1].in_channels
            kernel_size = backbone.classifier[1].kernel_size
            backbone.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size )
        elif 'cerranet' in self.model_name:
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(feature_dim, num_classes)

        self.backbone = backbone

    def extra_repr(self):
        s = ('backbone={model_name}, num_classes={num_classes}')
        return s.format(**self.__dict__)

    def forward(self, x):
        x = self.backbone(x)
        return x

