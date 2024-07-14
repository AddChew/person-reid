import math
import torch
import pickle

import warnings
import numpy as np
import os.path as osp

import torchvision.transforms as T

from torch import nn
from PIL import Image
from itertools import repeat
from functools import partial

from collections import OrderedDict
from torch.nn import functional as F
from collections import namedtuple, defaultdict


__all__ = [
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0'
]

pretrained_urls = {
    'osnet_x1_0':
    'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
    'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
    'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
    'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
    'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        IN=False,
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(cached_file)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


##########
# Instantiation
##########
def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model


def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # medium size (width x0.75)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_75')
    return model


def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # tiny size (width x0.5)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_5')
    return model


def osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # very tiny size (width x0.25)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_25')
    return model


def osnet_ibn_x1_0(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    # standard size (width x1.0) + IBN layer
    # Ref: Pan et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV, 2018.
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ibn_x1_0')
    return model


__model_factory = {
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
}


def _ntuple(n):

    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        return x

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

"""
Convolution
"""


def hook_convNd(m, x, y):
    k = torch.prod(torch.Tensor(m.kernel_size)).item()
    cin = m.in_channels
    flops_per_ele = k * cin # + (k*cin-1)
    if m.bias is not None:
        flops_per_ele += 1
    flops = flops_per_ele * y.numel() / m.groups
    return int(flops)


"""
Pooling
"""


def hook_maxpool1d(m, x, y):
    flops_per_ele = m.kernel_size - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_maxpool2d(m, x, y):
    k = _pair(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    # ops: compare
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_maxpool3d(m, x, y):
    k = _triple(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool1d(m, x, y):
    flops_per_ele = m.kernel_size
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool2d(m, x, y):
    k = _pair(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_avgpool3d(m, x, y):
    k = _triple(m.kernel_size)
    k = torch.prod(torch.Tensor(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool1d(m, x, y):
    x = x[0]
    out_size = m.output_size
    k = math.ceil(x.size(2) / out_size)
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool2d(m, x, y):
    x = x[0]
    out_size = _pair(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapmaxpool3d(m, x, y):
    x = x[0]
    out_size = _triple(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k - 1
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool1d(m, x, y):
    x = x[0]
    out_size = m.output_size
    k = math.ceil(x.size(2) / out_size)
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool2d(m, x, y):
    x = x[0]
    out_size = _pair(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


def hook_adapavgpool3d(m, x, y):
    x = x[0]
    out_size = _triple(m.output_size)
    k = torch.Tensor(list(x.size()[2:])) / torch.Tensor(out_size)
    k = torch.prod(torch.ceil(k)).item()
    flops_per_ele = k
    flops = flops_per_ele * y.numel()
    return int(flops)


"""
Non-linear activations
"""


def hook_relu(m, x, y):
    # eq: max(0, x)
    num_ele = y.numel()
    return int(num_ele)


def hook_leakyrelu(m, x, y):
    # eq: max(0, x) + negative_slope*min(0, x)
    num_ele = y.numel()
    flops = 3 * num_ele
    return int(flops)


"""
Normalization
"""


def hook_batchnormNd(m, x, y):
    num_ele = y.numel()
    flops = 2 * num_ele # mean and std
    if m.affine:
        flops += 2 * num_ele # gamma and beta
    return int(flops)


def hook_instancenormNd(m, x, y):
    return hook_batchnormNd(m, x, y)


def hook_groupnorm(m, x, y):
    return hook_batchnormNd(m, x, y)


def hook_layernorm(m, x, y):
    num_ele = y.numel()
    flops = 2 * num_ele # mean and std
    if m.elementwise_affine:
        flops += 2 * num_ele # gamma and beta
    return int(flops)


"""
Linear
"""


def hook_linear(m, x, y):
    flops_per_ele = m.in_features # + (m.in_features-1)
    if m.bias is not None:
        flops_per_ele += 1
    flops = flops_per_ele * y.numel()
    return int(flops)


__generic_flops_counter = {
    # Convolution
    'Conv1d': hook_convNd,
    'Conv2d': hook_convNd,
    'Conv3d': hook_convNd,
    # Pooling
    'MaxPool1d': hook_maxpool1d,
    'MaxPool2d': hook_maxpool2d,
    'MaxPool3d': hook_maxpool3d,
    'AvgPool1d': hook_avgpool1d,
    'AvgPool2d': hook_avgpool2d,
    'AvgPool3d': hook_avgpool3d,
    'AdaptiveMaxPool1d': hook_adapmaxpool1d,
    'AdaptiveMaxPool2d': hook_adapmaxpool2d,
    'AdaptiveMaxPool3d': hook_adapmaxpool3d,
    'AdaptiveAvgPool1d': hook_adapavgpool1d,
    'AdaptiveAvgPool2d': hook_adapavgpool2d,
    'AdaptiveAvgPool3d': hook_adapavgpool3d,
    # Non-linear activations
    'ReLU': hook_relu,
    'ReLU6': hook_relu,
    'LeakyReLU': hook_leakyrelu,
    # Normalization
    'BatchNorm1d': hook_batchnormNd,
    'BatchNorm2d': hook_batchnormNd,
    'BatchNorm3d': hook_batchnormNd,
    'InstanceNorm1d': hook_instancenormNd,
    'InstanceNorm2d': hook_instancenormNd,
    'InstanceNorm3d': hook_instancenormNd,
    'GroupNorm': hook_groupnorm,
    'LayerNorm': hook_layernorm,
    # Linear
    'Linear': hook_linear,
}

__conv_linear_flops_counter = {
    # Convolution
    'Conv1d': hook_convNd,
    'Conv2d': hook_convNd,
    'Conv3d': hook_convNd,
    # Linear
    'Linear': hook_linear,
}


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True
):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def _get_flops_counter(only_conv_linear):
    if only_conv_linear:
        return __conv_linear_flops_counter
    return __generic_flops_counter


def compute_model_complexity(
    model, input_size, verbose=False, only_conv_linear=True
):
    """Returns number of parameters and FLOPs.

    .. note::
        (1) this function only provides an estimate of the theoretical time complexity
        rather than the actual running time which depends on implementations and hardware,
        and (2) the FLOPs is only counted for layers that are used at test time. This means
        that redundant layers such as person ID classification layer will be ignored as it
        is discarded when doing feature extraction. Note that the inference graph depends on
        how you construct the computations in ``forward()``.

    Args:
        model (nn.Module): network model.
        input_size (tuple): input size, e.g. (1, 3, 256, 128).
        verbose (bool, optional): shows detailed complexity of
            each module. Default is False.
        only_conv_linear (bool, optional): only considers convolution
            and linear layers when counting flops. Default is True.
            If set to False, flops of all layers will be counted.

    Examples::
        >>> from torchreid import models, utils
        >>> model = models.build_model(name='resnet50', num_classes=1000)
        >>> num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)
    """
    registered_handles = []
    layer_list = []
    layer = namedtuple('layer', ['class_name', 'params', 'flops'])

    def _add_hooks(m):

        def _has_submodule(m):
            return len(list(m.children())) > 0

        def _hook(m, x, y):
            params = sum(p.numel() for p in m.parameters())
            class_name = str(m.__class__.__name__)
            flops_counter = _get_flops_counter(only_conv_linear)
            if class_name in flops_counter:
                flops = flops_counter[class_name](m, x, y)
            else:
                flops = 0
            layer_list.append(
                layer(class_name=class_name, params=params, flops=flops)
            )

        # only consider the very basic nn layer
        if _has_submodule(m):
            return

        handle = m.register_forward_hook(_hook)
        registered_handles.append(handle)

    default_train_mode = model.training

    model.eval().apply(_add_hooks)
    input = torch.rand(input_size)
    if next(model.parameters()).is_cuda:
        input = input.cuda()
    model(input) # forward

    for handle in registered_handles:
        handle.remove()

    model.train(default_train_mode)

    if verbose:
        per_module_params = defaultdict(list)
        per_module_flops = defaultdict(list)

    total_params, total_flops = 0, 0

    for layer in layer_list:
        total_params += layer.params
        total_flops += layer.flops
        if verbose:
            per_module_params[layer.class_name].append(layer.params)
            per_module_flops[layer.class_name].append(layer.flops)

    if verbose:
        num_udscore = 55
        print('  {}'.format('-' * num_udscore))
        print('  Model complexity with input size {}'.format(input_size))
        print('  {}'.format('-' * num_udscore))
        for class_name in per_module_params:
            params = int(np.sum(per_module_params[class_name]))
            flops = int(np.sum(per_module_flops[class_name]))
            print(
                '  {} (params={:,}, flops={:,})'.format(
                    class_name, params, flops
                )
            )
        print('  {}'.format('-' * num_udscore))
        print(
            '  Total (params={:,}, flops={:,})'.format(
                total_params, total_flops
            )
        )
        print('  {}'.format('-' * num_udscore))

    return total_params, total_flops


class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features