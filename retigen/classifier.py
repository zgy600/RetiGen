import logging
import torch
import torch.nn as nn
import torchvision.models as models
import modeling.model_manager as models_GDRBench
import os

class Classifier(nn.Module):
    def __init__(self, args, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None
        print(self.use_bottleneck)
        # print(self.use_bottleneck)
        # print('#########################################')
        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = models.__dict__[args.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = models.__dict__[args.arch](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
            bn = nn.BatchNorm1d(args.bottleneck_dim)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = args.bottleneck_dim

        self.fc = nn.Linear(self.output_dim, args.num_classes)

        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=args.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0


class Classifier_RetiGenBench(nn.Module):
    def __init__(self, args, net_path=None, classifier_path=None):
        super().__init__()
        self.args = args
        model = None
        self.network = models_GDRBench.get_net(args)
        self.classifier = models_GDRBench.get_classifier(self.network.out_features(), args)
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))
        
        model = models.__dict__[args.arch](pretrained=True)
        # modules = list(model.children())[:-1]
        # self.encoder = nn.Sequential(*modules)
        self._output_dim = model.fc.in_features
        if args.algorithm == 'MixStyleNet':
            print("MixStyleNet")
            self._output_dim = 512
        elif args.algorithm == 'CABNet':
            print("CABNet")
            self._output_dim = 1024
        #self._output_dim = 512
        self.encoder = self.network
        self.fc = self.classifier

        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=args.weight_norm_dim)
            

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        ##################################
        # feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, net_path, classifier_path):
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        checkpoint_net_path = torch.load(net_path, map_location="cpu")
        checkpoint_classifier_path = torch.load(classifier_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint_net_path["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        for name, param in checkpoint_classifier_path["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {net_path}; missing params: {msg.missing_keys}"
        )
    
    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        # if not self.use_bottleneck:
        #     backbone_params.extend(self.encoder.parameters())
        # # case 2)
        # else:
        #     resnet = self.encoder[0]
        #     for module in list(resnet.children())[:-1]:
        #         backbone_params.extend(module.parameters())
        #     # bottleneck fc + (bn) + classifier fc
        #     extra_params.extend(resnet.fc.parameters())
        #     extra_params.extend(self.encoder[1].parameters())
        #     extra_params.extend(self.fc.parameters())
            
        backbone_params.extend(self.encoder.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0

class Classifier_RetiGenBench_GREEN(nn.Module):
    def __init__(self, args, net_path=None, classifier_path=None):
        super().__init__()
        self.args = args
        model = None
        self.network = models_GDRBench.get_net(args)
        self.network.load_state_dict(torch.load(net_path))
        self.encoder_feats = self.network.cnn
        # A = torch.ones(32, 3, 224, 224)
        # print(self.encoder(A).shape)
        model = models.__dict__[args.arch](pretrained=True)
        # modules = list(model.children())[:-1]
        # self.encoder = nn.Sequential(*modules)
        self._output_dim = model.fc.in_features
        self.encoder = self.network
            

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder_feats(x)
        logits = self.encoder(x)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, net_path, classifier_path):
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        checkpoint_net_path = torch.load(net_path, map_location="cpu")
        checkpoint_classifier_path = torch.load(classifier_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint_net_path["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        for name, param in checkpoint_classifier_path["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {net_path}; missing params: {msg.missing_keys}"
        )
    
    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        backbone_params.extend(self.encoder.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        # return self.fc.weight.shape[0]
        return 5

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0
# class Classifier_RetiGenBench_GREEN(nn.Module):
#     def __init__(self, args, net_path=None, classifier_path=None):
#         super().__init__()
#         self.args = args
#         model = None
#         self.network = models_GDRBench.get_net(args)
#         self.network.load_state_dict(torch.load(net_path))
#         self.encoder_feats = self.network.cnn
#         # A = torch.ones(32, 3, 224, 224)
#         # print(self.encoder(A).shape)
#         model = models.__dict__[args.arch](pretrained=True)
#         # modules = list(model.children())[:-1]
#         # self.encoder = nn.Sequential(*modules)
#         self._output_dim = model.fc.in_features
#         self.encoder = self.network
            

#     def forward(self, x, return_feats=False):
#         # 1) encoder feature
#         feat = self.encoder_feats(x)
#         logits = self.encoder(x)

#         if return_feats:
#             return feat, logits
#         return logits

#     def load_from_checkpoint(self, net_path, classifier_path):
#         # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
#         checkpoint_net_path = torch.load(net_path, map_location="cpu")
#         checkpoint_classifier_path = torch.load(classifier_path, map_location="cpu")
#         state_dict = dict()
#         for name, param in checkpoint_net_path["state_dict"].items():
#             # get rid of 'module.' prefix brought by DDP
#             name = name.replace("module.", "")
#             state_dict[name] = param
#         for name, param in checkpoint_classifier_path["state_dict"].items():
#             # get rid of 'module.' prefix brought by DDP
#             name = name.replace("module.", "")
#             state_dict[name] = param
#         msg = self.load_state_dict(state_dict, strict=False)
#         logging.info(
#             f"Loaded from {net_path}; missing params: {msg.missing_keys}"
#         )
    
#     def get_params(self):
#         """
#         Backbone parameters use 1x lr; extra parameters use 10x lr.
#         """
#         backbone_params = []
#         extra_params = []
#         backbone_params.extend(self.encoder.parameters())

#         # exclude frozen params
#         backbone_params = [param for param in backbone_params if param.requires_grad]
#         extra_params = [param for param in extra_params if param.requires_grad]

#         return backbone_params, extra_params

#     @property
#     def num_classes(self):
#         # return self.fc.weight.shape[0]
#         return 5

#     @property
#     def output_dim(self):
#         return self._output_dim

#     @property
#     def use_bottleneck(self):
#         return self.args.bottleneck_dim > 0

#     @property
#     def use_weight_norm(self):
#         return self.args.weight_norm_dim >= 0

#     def __init__(self, args, net_path=None, classifier_path=None):
#         super().__init__()
#         self.args = args
#         model = None
#         self.network = models_GDRBench.get_net(args)
#         self.network.load_state_dict(torch.load(net_path))
        
#         model = models.__dict__[args.arch](pretrained=True)
#         # modules = list(model.children())[:-1]
#         # self.encoder = nn.Sequential(*modules)
#         self._output_dim = model.fc.in_features
#         self.encoder = self.network
            

#     def forward(self, x, return_feats=False):
#         # 1) encoder feature
#         feat = self.encoder(x)
#         logits = torch.flatten(feat, 1)
#         if return_feats:
#             return feat, logits
#         return logits

#     def load_from_checkpoint(self, net_path, classifier_path):
#         # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
#         checkpoint_net_path = torch.load(net_path, map_location="cpu")
#         checkpoint_classifier_path = torch.load(classifier_path, map_location="cpu")
#         state_dict = dict()
#         for name, param in checkpoint_net_path["state_dict"].items():
#             # get rid of 'module.' prefix brought by DDP
#             name = name.replace("module.", "")
#             state_dict[name] = param
#         for name, param in checkpoint_classifier_path["state_dict"].items():
#             # get rid of 'module.' prefix brought by DDP
#             name = name.replace("module.", "")
#             state_dict[name] = param
#         msg = self.load_state_dict(state_dict, strict=False)
#         logging.info(
#             f"Loaded from {net_path}; missing params: {msg.missing_keys}"
#         )
    
#     def get_params(self):
#         """
#         Backbone parameters use 1x lr; extra parameters use 10x lr.
#         """
#         backbone_params = []
#         extra_params = []
#         backbone_params.extend(self.encoder.parameters())

#         # exclude frozen params
#         backbone_params = [param for param in backbone_params if param.requires_grad]
#         extra_params = [param for param in extra_params if param.requires_grad]

#         return backbone_params, extra_params

#     @property
#     def num_classes(self):
#         # return self.fc.weight.shape[0]
#         return 5

#     @property
#     def output_dim(self):
#         return self._output_dim

#     @property
#     def use_bottleneck(self):
#         return self.args.bottleneck_dim > 0

#     @property
#     def use_weight_norm(self):
#         return self.args.weight_norm_dim >= 0

class OriginModel(nn.Module):
    def __init__(self, args, log_path):
        super(OriginModel, self).__init__()
        # 在这里定义你的模型结构，可以使用已有的层（例如线性层、卷积层等）组成你的网络
        # 使用cfg来配置模型的参数，根据你的需求进行配置
        self.net = models_GDRBench.get_net(args)
        self.classifier = models_GDRBench.get_classifier(self.net.out_features(), args)
        
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
