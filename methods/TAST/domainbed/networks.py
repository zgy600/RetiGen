# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.model_zoo as model_zoo

from domainbed.lib import misc
from domainbed.lib import wide_resnet
from domainbed.lib import big_transfer
# from domainbed.lib import vision_transformer
# from domainbed.lib import mlp_mixer


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

class Backbone(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# class SqueezeLastTwo(nn.Module):
#     """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
#     def __init__(self):
#         super(SqueezeLastTwo, self).__init__()

#     def forward(self, x):
#         return x.view(x.shape[0], x.shape[1])


# class MLP(nn.Module):
#     """Just  an MLP"""
#     def __init__(self, n_inputs, n_outputs, hparams):
#         super(MLP, self).__init__()
#         self.input = nn.Linear(n_inputs, hparams['mlp_width'])
#         self.dropout = nn.Dropout(hparams['mlp_dropout'])
#         self.hiddens = nn.ModuleList([
#             nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
#             for _ in range(hparams['mlp_depth']-2)])
#         self.output = nn.Linear(hparams['mlp_width'], n_outputs)
#         self.n_outputs = n_outputs

#     def forward(self, x):
#         x = self.input(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         for hidden in self.hiddens:
#             x = hidden(x)
#             x = self.dropout(x)
#             x = F.relu(x)
#         x = self.output(x)
#         return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['backbone'] == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet50':
            # self.network = torchvision.models.resnet50(pretrained=True)
            self.network = resnet50(pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet18-BN':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet50-BN':
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = False

        if self.disable_bn:
            self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # print(self.network)
        
        # del self.network.fc
        self.network.fc = Identity()
        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()
 
    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    # pretrain_dict = torch.load(model_url)
    model.load_state_dict(pretrain_dict, strict=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet50(pretrained=True, **kwargs):
    model = ResNet_RetiGen(block=Bottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model

class ResNet_RetiGen(Backbone):
    
    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x)
        return self.layer4(x)

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)
        #return v

# class MNIST_CNN(nn.Module):
#     """
#     Hand-tuned architecture for MNIST.
#     Weirdness I've noticed so far with this architecture:
#     - adding a linear layer after the mean-pool in features hurts
#         RotatedMNIST-100 generalization severely.
#     """
#     n_outputs = 128

#     def __init__(self, input_shape):
#         super(MNIST_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

#         self.bn0 = nn.GroupNorm(8, 64)
#         self.bn1 = nn.GroupNorm(8, 128)
#         self.bn2 = nn.GroupNorm(8, 128)
#         self.bn3 = nn.GroupNorm(8, 128)

#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.squeezeLastTwo = SqueezeLastTwo()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.bn0(x)

#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.bn1(x)

#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.bn2(x)

#         x = self.conv4(x)
#         x = F.relu(x)
#         x = self.bn3(x)

#         x = self.avgpool(x)
#         x = self.squeezeLastTwo(x)
#         return x


# class ContextNet(nn.Module):
#     def __init__(self, input_shape):
#         super(ContextNet, self).__init__()

#         # Keep same dimensions
#         padding = (5 - 1) // 2
#         self.context_net = nn.Sequential(
#             nn.Conv2d(input_shape[0], 64, 5, padding=padding),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 5, padding=padding),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, 5, padding=padding),
#         )

#     def forward(self, x):
#         return self.context_net(x)
# def Featurizer(input_shape, hparams):
#     """Auto-select an appropriate featurizer for the given input shape."""
#     if len(input_shape) == 1:
#         return MLP(input_shape[0], 128, hparams)
#     elif input_shape[1:3] == (28, 28):
#         return MNIST_CNN(input_shape)
#     elif input_shape[1:3] == (32, 32):
#         return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
#     elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['resnet50', 'resnet18', 'resnet50-BN', 'resnet18-BN']:
#         return ResNet(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'ViT-' in hparams['backbone']:
#         return vision_transformer.ViT2(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['B_16', 'B_32', 'L_16', 'L_32']:
#         return vision_transformer.ViT(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'dino' in hparams['backbone']:
#         return vision_transformer.DINO(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'DeiT' in hparams['backbone']:
#         return vision_transformer.DeiT(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'HViT' in hparams['backbone']:
#         return vision_transformer.HybridViT(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'Mixer' in hparams['backbone']:
#         return mlp_mixer.MLPMixer(input_shape, hparams)
#     elif input_shape[1:3] == (224, 224) and 'BiT' in hparams['backbone']:
#         return big_transfer.BiT(input_shape, hparams)
#     else:
#         raise NotImplementedError

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if input_shape[1:3] == (224, 224) and hparams['backbone'] in ['resnet50', 'resnet18', 'resnet50-BN', 'resnet18-BN']:
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
