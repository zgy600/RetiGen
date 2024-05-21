# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.algs.base import Algorithm
from modeling import model_manager as models

import os
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        # self.featurizer = get_fea(args)
        # self.classifier = common_network.feat_classifier(
        #     args.num_classes, self.featurizer.in_features, args.classifier)
        self.featurizer = models.get_backbone_resnet50()
        self.classifier = models.get_classifier(self.featurizer.out_features())

        self.network = nn.Sequential(
            self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        print(net_path)
        print(classifier_path)
        self.featurizer.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

# class ERM(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, args):
#         super(ERM, self).__init__(args)
#         self.featurizer = get_fea(args)
#         self.classifier = common_network.feat_classifier(
#             args.num_classes, self.featurizer.in_features, args.classifier)

#         self.network = nn.Sequential(
#             self.featurizer, self.classifier)

#     def update(self, minibatches, opt, sch):
#         all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#         all_y = torch.cat([data[1].cuda().long() for data in minibatches])
#         loss = F.cross_entropy(self.predict(all_x), all_y)

#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         if sch:
#             sch.step()
#         return {'class': loss.item()}

#     def predict(self, x):
#         return self.network(x)
