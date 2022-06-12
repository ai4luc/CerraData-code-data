'''
pytorch Model architecture and its converted weights from keras was obtained from https://github.com/jurandy-almeida/cerranet

Minor changes was made in order to get compatibility with the running code.
'''

import os
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from collections import OrderedDict


__all__ = ['CerraNet', 'cerranet']


model_urls = {
    'cerranet': 'file://' +
           os.path.dirname(os.path.abspath(__file__)) + '/'
           'cerranet-95d1d357.pth'
}


class CerraNet(nn.Module):

    def __init__(self, num_classes=4):
        super(CerraNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop1', nn.Dropout(0.15)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop2', nn.Dropout(0.15)),
            ('conv3', nn.Conv2d(64, 128, kernel_size=3)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop3', nn.Dropout(0.15)),
            ('conv4', nn.Conv2d(128, 128, kernel_size=3)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop4', nn.Dropout(0.15)),
            ('conv5', nn.Conv2d(128, 256, kernel_size=3)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop5', nn.Dropout(0.15)),
            ('conv6', nn.Conv2d(256, 256, kernel_size=3)),
            ('relu6', nn.ReLU(inplace=True)),
            ('pool6', nn.AvgPool2d(kernel_size=(2, 2))),
            ('drop6', nn.Dropout(0.15)),
        ]))
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(256 * 2 * 2, 256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout(0.15)),
            ('fc8', nn.Linear(256, 128)),
            ('relu8', nn.ReLU(inplace=True)),
            ('drop8', nn.Dropout(0.15)),
            ('fc9', nn.Linear(128, num_classes)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def cerranet(pretrained=False, progress=True, **kwargs):
    r"""CerraNet model architecture from the
    `"CerraNet: a deep convolutional neural network for classifying land use and land cover on Cerrado biome tocantinense" <https://drive.google.com/file/d/1JnN52C8yZKwN-5XA6qSiCsCygh1-0vvZ/view>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Sports1M
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = CerraNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['cerranet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
