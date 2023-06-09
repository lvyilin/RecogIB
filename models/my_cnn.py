import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_cfg = {
    'backbone': [{"type": "conv", "filters": 128, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu",
                  "batch_norm": True},
                 {"type": "conv", "filters": 128, "kernel_size": 4, "stride": 2, "padding": 1, "activation": "relu",
                  "batch_norm": True},
                 {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 2, "padding": 0, "activation": "relu",
                  "batch_norm": True},
                 {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 0,
                  "activation": "relu",
                  "batch_norm": True},
                 {"type": "avgpool"},
                 {"type": "flatten"},
                 {"type": "fc", "dim": 1024, "activation": "relu", "dropout": 0.0},
                 {"type": "fc", "dim": 512, "activation": "relu", "dropout": 0.0}],
    'classifier': [{"type": "fc"}],
}


class Model(nn.Module):
    def __init__(
            self, backbone: nn.Module, classifier: nn.Module, init_weights=True, reparametrize='none') -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.feat_size = classifier[0].in_features
        self.reparametrize = reparametrize
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        self.backbone[16].register_forward_hook(self.save_pre_feature)
        
    def save_pre_feature(self, module, input, output):
        self.pre_feature = output
        

    def forward(self, x, with_feat=False, with_pre_feat=False):
        feature = self.backbone(x)
        if self.reparametrize == 'vib':
            mu, std = torch.chunk(feature, 2, dim=1)
            std = F.softplus(std - 5)
            if self.training:
                feature = (mu, std)
                logits = self.classifier(mu + std * torch.randn_like(std))
            else:
                feature = mu
                logits = self.classifier(mu)
        elif self.reparametrize == 'nib':
            mu = feature
            std = math.exp(-0.5)
            if self.training:
                feature = (mu, std)
                logits = self.classifier(mu + std * torch.randn_like(mu))
            else:
                logits = self.classifier(mu)
        else:
            logits = self.classifier(feature)
            
        if not with_feat and not with_pre_feat:
            return logits
        ret = [logits]
        if with_feat:
            ret.append(feature)
        if with_pre_feat:
            ret.append(self.pre_feature)
        return ret


def get_network(cfg, input_channels: int = 3, reparametrize: str = 'none'):
    backbone = get_sequential(cfg['backbone'], input_channels, reparametrize)
    feature_dim = cfg['backbone'][-1]['dim']
    classifier = get_sequential(cfg['classifier'], feature_dim, False)
    return Model(backbone, classifier, reparametrize=reparametrize)


def get_sequential(cfg, input_channels: int, reparametrize: str):
    layers = []
    in_channels = input_channels
    for i, v in enumerate(cfg):
        if v['type'] == "conv":
            conv2d = nn.Conv2d(in_channels, v['filters'], v['kernel_size'], stride=v['stride'], padding=v['padding'])
            if v['batch_norm']:
                layers += [conv2d, nn.BatchNorm2d(v['filters']), get_activation(v['activation'])]
            else:
                layers += [conv2d, get_activation(v['activation'])]
            in_channels = v['filters']
        elif v['type'] == "fc":
            if reparametrize == 'vib' and i == len(cfg) - 1:
                fc = nn.Linear(in_channels, v['dim'] * 2)
            else:
                fc = nn.Linear(in_channels, v['dim'])
            layers.append(fc)
            if 'activation' in v:
                layers.append(get_activation(v['activation']))
            if 'dropout' in v:
                layers.append(nn.Dropout(v['dropout']))

            in_channels = v['dim']
        elif v['type'] == 'flatten':
            layers.append(nn.Flatten(1))
        elif v['type'] == 'avgpool':
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            raise NotImplementedError

    return nn.Sequential(*layers)


def get_activation(act, inplace=True):
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'tanh':
        return nn.Tanh()
    raise NotImplementedError


def CNN(num_classes, input_channels, dropout_rate, reparametrize='none'):
    cfg = _cfg
    cfg['classifier'][-1]['dim'] = num_classes
    if dropout_rate > 0.0:
        for layers in cfg.values():
            for la in layers:
                if "dropout" in la:
                    la['dropout'] = dropout_rate

    return get_network(cfg, input_channels, reparametrize)
