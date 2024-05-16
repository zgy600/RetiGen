
from .resnet import resnet18, resnet50, resnet101
from .nets import *
import torch
import os
import ttach as tta

def get_net(cfg):
    if cfg.algorithm == 'ERM' or cfg.algorithm == 'GDRNet':
        net = get_backbone(cfg)
    elif cfg.algorithm == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.algorithm == 'CABNet':
        net = CABNet(cfg)
    elif cfg.algorithm == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.algorithm == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.algorithm == 'Fishr' or cfg.algorithm == 'DRGen' \
        or cfg.algorithm == 'FishrAUG' or cfg.algorithm == 'FishrCONAUG' \
        or cfg.algorithm == 'FishrCONAUG1' or cfg.algorithm == 'FishrCONAUG2'\
        or cfg.algorithm == 'FishrCONAUG3':
        net = FishrNet(cfg)
    elif cfg.algorithm == 'FishrDSU' or cfg.algorithm == 'FishrDSUAUG':
        net = FishrDSUNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net

def get_backbone(cfg):
    if cfg.backbone == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.backbone == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.backbone == 'resnet101':
        model = resnet101(pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model

def get_classifier(out_feature_size, cfg):
    # return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)
    return torch.nn.Linear(out_feature_size, cfg.num_classes)

def get_classifier_tta(cfg, log_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = OriginModel(cfg, log_path)
    model.eval()
    model = model.to(device)
    
    # defined 2 * 2 * 3 * 3 = 36 augmentations !
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),        
        ]
    )   
    # tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(1, 1))
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())
    # tta_model = tta.ClassificationTTAWrapper(model, transforms)
    return tta_model

class OriginModel(nn.Module):
    def __init__(self, cfg, log_path):
        super(OriginModel, self).__init__()
        # 在这里定义你的模型结构，可以使用已有的层（例如线性层、卷积层等）组成你的网络
        # 使用cfg来配置模型的参数，根据你的需求进行配置
        self.net = get_net(cfg)
        self.classifier = get_classifier(self.net.out_features(), cfg)
        
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        
        self.net.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def forward(self, x):
        # 在这里定义前向传播逻辑，即如何通过输入x计算模型的输出
        # 可以使用之前定义的网络层（self.net）和分类器（self.classifier）
        x = self.net(x)
        x = self.classifier(x)
        return x
