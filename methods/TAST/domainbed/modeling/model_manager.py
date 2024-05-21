
from .resnet import resnet18, resnet50, resnet101
from .nets import *
import torch
import os
# import ttach as tta
# import tent
import torch.optim as optim

def get_net(cfg):
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.ALGORITHM == 'CABNet':
        net = CABNet(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen' \
        or cfg.ALGORITHM == 'FishrAUG' or cfg.ALGORITHM == 'FishrCONAUG' \
        or cfg.ALGORITHM == 'FishrCONAUG1' or cfg.ALGORITHM == 'FishrCONAUG2'\
        or cfg.ALGORITHM == 'FishrCONAUG3':
        net = FishrNet(cfg)
    elif cfg.ALGORITHM == 'FishrDSU' or cfg.ALGORITHM == 'FishrDSUAUG':
        net = FishrDSUNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net

def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model

def get_backbone_resnet50():
    model = resnet50(pretrained=True)
    return model

def get_classifier(out_feature_size):
    return torch.nn.Linear(out_feature_size, 5)

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

def get_tent(cfg, log_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = OriginModel(cfg, log_path)
    tent_model = setup_tent(model)
    # model.eval()
    tent_model = tent_model.to(device)
    return model

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

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    METHOD='Adam'
    if METHOD == 'Adam':
        return optim.Adam(params,
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    weight_decay=0.)
    else:
        raise NotImplementedError