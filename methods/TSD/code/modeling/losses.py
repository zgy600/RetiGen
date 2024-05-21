"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Our loss function
class DahLoss(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))
        
        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class DahLoss2(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss2, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss2(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            
        
        # if (features_new.shape[0]==16):
        #     sequence_ori = np.array([1, 2, 3, 4])
        #     shuffled_sequences = []
        #     # 打乱4次原始序列
        #     for i in range(4):
        #         sequence = np.copy(sequence_ori)
        #         np.random.shuffle(sequence)
        #         # 根据组数添加不同的偏移
        #         if i == 1:
        #             sequence += 4
        #         elif i == 2:
        #             sequence += 8
        #         elif i == 3:
        #             sequence += 12
        #         shuffled_sequences.append(sequence)

        #     # 将4次的结果拼接成一个16个数字的序列
        #     result_sequence = np.concatenate(shuffled_sequences)
        #     reshuffled_features_new = features_new[result_sequence - 1]
        #     features_new = reshuffled_features_new
        
        # if (features_new.shape[0]==16):
        #     indices_to_replace = [0, 4, 8, 12] # features_new的V1
        #     indices_to_copy = [1, 5, 9, 13] # features_ori的disc-centered
        #     # 替换指定行
        #     for i in range(len(indices_to_replace)):
        #         features_new[indices_to_replace[i], :] = features_ori[indices_to_copy[i], :]

        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)
        # if (features_new.shape[0]==16):      
        #     labels = torch.tensor([1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12])
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        # loss = (1 - self.alpha) * loss_sup
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class DahLoss_mv(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss_mv, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss_mv(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains, split_tensors, mv_list_index):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            
        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels, split_tensors, mv_list_index, mask=None))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        # loss = (1 - self.alpha) * loss_sup
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class DahLoss25(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss25, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            
        
        
        if (features_new.shape[0]==16):
            indices_to_replace = [0, 4, 8, 12] # features_new的V1
            indices_to_copy = [1, 5, 9, 13] # features_ori的disc-centered
            # 替换指定行
            for i in range(len(indices_to_replace)):
                features_new[indices_to_replace[i], :] = features_ori[indices_to_copy[i], :]

        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)
        # if (features_new.shape[0]==16):      
        #     labels = torch.tensor([1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12])
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        # loss = (1 - self.alpha) * loss_sup
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class DahLoss50(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss50, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            
        
        if (features_new.shape[0]==16):
            indices_to_replace = [0, 4, 8, 12, 2, 6, 10, 14] # features_new的V1 V3
            indices_to_copy = [1, 5, 9, 13, 1, 5, 9, 13] # features_ori的disc-centered
            # 替换指定行
            for i in range(len(indices_to_replace)):
                features_new[indices_to_replace[i], :] = features_ori[indices_to_copy[i], :]

        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)
        # if (features_new.shape[0]==16):      
        #     labels = torch.tensor([1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12])
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        # loss = (1 - self.alpha) * loss_sup
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha
    
class DahLoss75(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(DahLoss75, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593,
                                'MFIDDR_1view': 8613,
                                'MFIDDR_4views': 34452,
                                'MFIDDR_4viewss': 8612,
                                'DRTiD_1view': 1550,
                                'DRTiD_2views': 3100}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62],
                                'MFIDDR_1view': [5248, 1832, 752, 621, 160],
                                'MFIDDR_4views': [20992, 7328, 3008, 2484, 640],
                                'MFIDDR_4viewss': [5248, 1832, 752, 620, 160],
                                'DRTiD_1view': [747, 140, 406, 199, 58],
                                'DRTiD_2views': [1494, 280, 812, 398, 116]}

        self.domain_prob, self.label_prob = self.get_domain_label_prob()
        self.domain_prob, self.label_prob = self.multinomial_soomthing(self.domain_prob, self.label_prob, self.beta)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob.cuda(), label_prob.cuda()

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains).cuda()
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels).cuda()
        class_weight = 1 / class_prob

        return domain_weight, class_weight
                            
    def forward(self, output, features, labels, domains):
        
        domain_weight, class_weight = self.get_weights(labels, domains)

        loss_dict = {}

        features_ori, features_new = features

        loss_sup = 0

        for op_item in output:
            loss_sup += self.SupLoss(op_item, labels)            
        
        if (features_new.shape[0]==16):
            indices_to_replace = [0, 4, 8, 12, 2, 6, 10, 14, 3, 7, 11, 15] # features_new的V1 V3 V4
            indices_to_copy = [1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13] # features_ori的disc-centered
            # 替换指定行
            for i in range(len(indices_to_replace)):
                features_new[indices_to_replace[i], :] = features_ori[indices_to_copy[i], :]

        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)
        # if (features_new.shape[0]==16):      
        #     labels = torch.tensor([1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12])
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        # loss = (1 - self.alpha) * loss_sup
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha


# Contrastive loss function
class ConLoss1(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(ConLoss1, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature 
        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        
                            
    def forward(self, output, features, labels, domains):
        
        loss_dict = {}

        features_ori, features_new = features         
        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)    
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
         
        # loss =  self.alpha * loss_unsup / self.scaling_factor
        loss =  self.alpha * loss_unsup / self.scaling_factor
        
        loss_dict['loss'] = loss.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class ConLoss2(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(ConLoss2, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature 

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
                         
    def forward(self, output, features, labels, domains):
        
        loss_dict = {}
        features_ori, features_new = features         
        
        if (features_new.shape[0]==16):
            sequence_ori = np.array([1, 2, 3, 4])
            shuffled_sequences = []
            # 打乱4次原始序列
            for i in range(4):
                sequence = np.copy(sequence_ori)
                np.random.shuffle(sequence)
                # 根据组数添加不同的偏移
                if i == 1:
                    sequence += 4
                elif i == 2:
                    sequence += 8
                elif i == 3:
                    sequence += 12
                shuffled_sequences.append(sequence)

            # 将4次的结果拼接成一个16个数字的序列
            result_sequence = np.concatenate(shuffled_sequences)
            reshuffled_features_new = features_new[result_sequence - 1]
            features_new = reshuffled_features_new
        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)    
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        
        # loss_sup = torch.mean(loss_sup * class_weight * domain_weight) / (torch.mean(domain_weight) * torch.mean(class_weight))

        # loss = (1 - self.alpha) * loss_sup + self.alpha * loss_unsup / self.scaling_factor
        
        # loss =  self.alpha * loss_unsup / self.scaling_factor
        loss =  self.alpha * loss_unsup / self.scaling_factor
        
        loss_dict['loss'] = loss.item()
        # loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha
    
class ConLoss3(nn.Module):
    def __init__(self, max_iteration, training_domains, beta = 0.8, scaling_factor = 4, alpha = 1, temperature = 0.07) -> None:
        super(ConLoss3, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature 

        self.UnsupLoss = SupConLoss2(temperature = self.temperature, reduction='none')

                            
    def forward(self, output, features, labels, domains):
        
        loss_dict = {}

        features_ori, features_new = features         
        
        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)    
        
        # loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        loss_unsup = torch.mean(self.UnsupLoss(features_multi, labels))
        
        # loss =  self.alpha * loss_unsup / self.scaling_factor
        loss =  self.alpha * loss_unsup / self.scaling_factor
        
        loss_dict['loss'] = loss.item()
        # loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        return self.alpha

class SupConLoss(nn.Module): # Unsupervised contrastive learning
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss

class SupConLoss2(nn.Module):# Supervised contrastive learning
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(features.shape)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # print(anchor_dot_contrast.shape)
        # print(anchor_dot_contrast)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(logits_mask.shape)
        # print(logits_mask)
        mask = mask * logits_mask
        # print(mask[16,:])
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss

class SupConLoss_mv(nn.Module):# multi-views Supervised contrastive learning
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss_mv, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels, split_tensors, mv_list_index, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(features.shape)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        if len(split_tensors)>0:
            merged_tensor = torch.cat(split_tensors, dim=0)
            merged_tensor = F.normalize(merged_tensor, p=2, dim=1)
            mv_num_list = [tensor.shape[0] for tensor in split_tensors]
            mv_dot_contrast = torch.div(
                torch.matmul(anchor_feature, merged_tensor.T),
                self.temperature)
            mv_mask = torch.zeros_like(mv_dot_contrast, dtype=torch.float32).to(device)
            current_index = 0
            for i, mv_index in enumerate(mv_list_index):
                mv_num = mv_num_list[i]
                mv_mask[mv_index, current_index:current_index+mv_num] = 1
                mv_mask[mv_index + batch_size, current_index:current_index+mv_num] = 1
                current_index += mv_num
            mv_logits = mv_dot_contrast - logits_max.detach()
            mv_logits = mv_logits * mv_mask
            exp_logits_mv = torch.exp(mv_logits)
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(logits_mask.shape)
        # print(logits_mask)
        mask = mask * logits_mask
        # print(mask[16,:])
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if len(split_tensors)>0:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) - torch.log(exp_logits_mv.sum(1, keepdim=True))
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss

class SupConLoss3(nn.Module):# Supervised contrastive learning
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss3, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # 2
        # print(features.shape) # [16, 2, 2048]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [32, 2048]
        # print(contrast_feature.shape)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # [16, 2048]
            print(anchor_feature.shape)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # [32, 2048]
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # print(anchor_dot_contrast.shape)
        # print(anchor_dot_contrast)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(mask.shape)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(logits_mask.shape)
        # print(logits_mask)
        mask = mask * logits_mask # 对角线清零
        # print(mask[16,:])
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask #对角线清零的exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob.shape)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss
