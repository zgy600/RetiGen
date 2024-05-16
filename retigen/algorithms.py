"""
This code collected some methods from DomainBed (https://github.com/facebookresearch/DomainBed) and other SOTA methods.
"""
import os, collections, logging
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, collections, logging
from collections import OrderedDict

import utilsGDRBench.misc as misc
from utilsGDRBench.validate import algorithm_validate, algorithm_validate_test, algorithm_validate_tta_test, algorithm_generate_pseudo_labels, algorithm_generate_pseudo_labels_MVLCE, algorithm_validate_ttamv_test
import modeling.model_manager as models
from modeling.losses import DahLoss
from modeling.nets import LossValley, AveragedModel
from utilsGDRBench.data_manager import get_post_FundusAug, get_pre_FundusAug

from backpack import backpack, extend
from backpack.extensions import BatchGrad

import ttach as tta
from PIL import Image
import glob
import re

ALGORITHMS = [
    'ERM',
    'GDRNet',
    'GDRNetTTA',
    'GREEN',
    'CABNet',
    'MixupNet',
    'MixStyleNet',
    'Fishr',
    'DRGen'
    ]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - validate()
    - validate_tta()
    - save_model()
    - renew_model()
    - predict()
    - predict()
    """
    def __init__(self, num_classes, cfg):
        super(Algorithm, self).__init__()
        self.cfg = cfg
        self.epoch = 0

    def update(self, minibatches):
        raise NotImplementedError
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return epoch
    
    def validate(self, val_loader, test_loader, writer):
        raise NotImplementedError
    
    def validate_tta(self, cfg, val_loader, test_loader, writer):
        raise NotImplementedError
    
    def validate_tta_test(self, cfg, val_loader, test_loader):
        raise NotImplementedError
    
    def generate_pseudo_labels(self, test_loader):
        val_auc = -1
        test_auc = -1
        MVLCE = False
        if MVLCE:
            test_auc, test_acc, test_f1, test_loss = algorithm_generate_pseudo_labels_MVLCE(self, test_loader, self.epoch, 'test', self.cfg)
        else:
            test_auc, test_acc, test_f1, test_loss = algorithm_generate_pseudo_labels(self, test_loader, self.epoch, 'test', self.cfg)
        print("AUC:{:.10f} ACC:{:.10f} F1:{:.10f} Loss:{:.10f}".format(test_auc, test_acc, test_f1, test_loss))  
        return val_auc, test_auc
    
    def validate_test(self, val_loader, test_loader):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            # val_auc, val_loss = algorithm_validate_test(self, val_loader, self.epoch, 'val')
            test_auc, test_acc, test_f1, test_loss = algorithm_validate_test(self, test_loader, self.epoch, 'test')
            # test_auc, test_acc, test_f1, test_loss = algorithm_validate_test(self, test_loader, self.epoch, 'test', self.cfg)
            # test_auc, test_acc, test_f1, test_loss = algorithm_validate_ttamv_test(self, test_loader, self.epoch, 'test', self.cfg)
            # test_auc, test_acc, test_f1, test_loss = algorithm_generate_pseudo_labels(self, test_loader, self.epoch, 'test', self.cfg)
            print("AUC:{:.10f} ACC:{:.10f} F1:{:.10f} Loss:{:.10f}".format(test_auc, test_acc, test_f1, test_loss))

            if self.epoch == self.cfg.data.EPOCHS:
                self.epoch += 1
        else:
            # test_auc, test_loss = algorithm_validate_test(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            test_auc, test_loss = algorithm_validate_test(self, test_loader, self.cfg.data.EPOCHS + self.cfg.data.EPOCHS, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc
    
    def save_model(self, log_path):
        raise NotImplementedError
    
    def renew_model(self, log_path):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError
    
    def predict_tta(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, num_classes, cfg):
        super(ERM, self).__init__(num_classes, cfg)
        
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)
        self.classifier_tta = None
        self.tent_model = None
        
        self.optimizer = torch.optim.SGD(
            [{"params":self.network.parameters()},
            {"params":self.classifier.parameters()}],
            lr = cfg.data.LEARNING_RATE,
            momentum = cfg.data.MOMENTUM,
            weight_decay = cfg.data.WEIGHT_DECAY,
            nesterov=True)

    def update(self, minibatch):
        image, mask, label, domain, img_path = minibatch
        self.optimizer.zero_grad()

        features = self.network(image)
        output = self.classifier(features)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.data.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.data.EPOCHS + self.cfg.data.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc
    
    def validate_test(self, val_loader, test_loader):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            # val_auc, val_loss = algorithm_validate_test(self, val_loader, self.epoch, 'val')
            test_auc, test_acc, test_f1, test_loss = algorithm_validate_test(self, test_loader, self.epoch, 'test')
            # test_auc, test_acc, test_f1, test_loss = algorithm_validate_test(self, test_loader, self.epoch, 'test', self.cfg)
            # test_auc, test_acc, test_f1, test_loss = algorithm_validate_ttamv_test(self, test_loader, self.epoch, 'test', self.cfg)
            # test_auc, test_acc, test_f1, test_loss = algorithm_generate_pseudo_labels(self, test_loader, self.epoch, 'test', self.cfg)
            print("AUC:{:.10f} ACC:{:.10f} F1:{:.10f} Loss:{:.10f}".format(test_auc, test_acc, test_f1, test_loss))

            if self.epoch == self.cfg.data.EPOCHS:
                self.epoch += 1
        else:
            # test_auc, test_loss = algorithm_validate_test(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            test_auc, test_loss = algorithm_validate_test(self, test_loader, self.cfg.data.EPOCHS + self.cfg.data.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc
    
    
    def validate_tta_test(self, cfg, val_loader, test_loader):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # val_auc, val_loss = algorithm_validate_test(self, val_loader, self.epoch, 'val')
            # test_auc, test_loss = algorithm_validate_ttach_test(self, test_loader, self.epoch, 'test')
            test_auc, test_loss = algorithm_validate_tta_topK_test(self, cfg.data.source_domains, test_loader, self.epoch, 'test')
            
            if self.epoch == self.cfg.EPOCHS:
                self.epoch += 1
        else:
            # test_auc, test_loss = algorithm_validate_ttach_test(self, test_loader, self.cfg.EPOCHS + self.cfg.VAL_EPOCH, 'test')
            test_auc, test_loss = algorithm_validate_tta_topK_test(self, cfg.data.source_domains, test_loader, self.cfg.data.EPOCHS + self.cfg.data.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
                
        return val_auc, test_auc
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))

    def predict(self, x):
        return self.classifier(self.network(x))
    
    def renew_model_tta(self, cfg, log_path):
        self.classifier_tta = models.get_classifier_tta(cfg, log_path)
        
    def renew_model_tent(self, cfg, log_path):
        self.tent_model = models.get_tent(cfg, log_path)
    
    def predict_tent(self, x):
        A = self.tent_model(x)
        return self.tent_model(x)
    
    def predict_tta(self, x):
        # A = self.classifier_tta(x)
        # print(A)
        # print(type(A))
        A  = self.classifier_tta(x)
        return self.classifier_tta(x)
    
    def predict_tta_Tau_normalized(self, x):
        weights = self.classifier.weight.cpu()
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()
        p = 0
        for i in range(weights.size(0)):
            ws[i] = ws[i] / torch.pow(normB[i], p)
        ws = ws.to(self.classifier.weight.data.device)
        self.classifier.weight.data = ws
        return self.classifier(self.network(x))
        
    def pnorm(weights, p):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()
        for i in range(weights.size(0)):
            ws[i] = ws[i] / torch.pow(normB[i], p)
        return ws

# Our method
class GDRNet(ERM):
    def __init__(self, num_classes, cfg):
        super(GDRNet, self).__init__(num_classes, cfg)
        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.data.GDRNET.BETA, max_iteration = cfg.data.EPOCHS, \
                                training_domains = cfg.data.source_domains, temperature = cfg.data.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.data.GDRNET.SCALING_FACTOR)
                                    
    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori
        # return img_tensor_new, img_tensor
    
    def update(self, minibatch):
        
        image, mask, label, domain, img_path = minibatch
        
        self.optimizer.zero_grad()
        
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_new = self.classifier(features_new)
    
        loss, loss_dict_iter = self.criterion([output_new], [features_ori, features_new], label, domain)
        
        loss.backward()
        self.optimizer.step()

        return loss_dict_iter
    
    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)

class GREEN(Algorithm):
    def __init__(self, num_classes, cfg):
        super(GREEN, self).__init__(num_classes, cfg)
        self.network = models.get_net(cfg)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr = cfg.data.LEARNING_RATE,
            momentum = cfg.data.MOMENTUM,
            weight_decay = cfg.data.WEIGHT_DECAY,
            nesterov=True)
    
    def update(self, minibatch):
        image, mask, label, domain, img_path = minibatch
        self.optimizer.zero_grad()

        output = self.network(image)
        loss = F.cross_entropy(output, label)
        
        loss.backward()
        self.optimizer.step()
        return {'loss':loss}
    
    def validate(self, val_loader, test_loader, writer):
        val_auc = -1
        test_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            val_auc, val_loss = algorithm_validate(self, val_loader, writer, self.epoch, 'val')
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.epoch, 'test')
            if self.epoch == self.cfg.data.EPOCHS:
                self.epoch += 1
        else:
            test_auc, test_loss = algorithm_validate(self, test_loader, writer, self.cfg.data.EPOCHS + self.cfg.data.VAL_EPOCH, 'test')
            logging.info('Best performance on test domain(s): {}'.format(test_auc))
            
        return val_auc, test_auc
    
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        self.network.load_state_dict(torch.load(net_path))
    
    def predict(self, x):
        return self.network(x)
    
class CABNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(CABNet, self).__init__(num_classes, cfg)
        
class MixStyleNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixStyleNet, self).__init__(num_classes, cfg)
        
class MixupNet(ERM):
    
    def __init__(self, num_classes, cfg):
        super(MixupNet, self).__init__(num_classes, cfg)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
    
    def update(self, minibatch, env_feats=None):
        image, mask, label, domain, img_path = minibatch
        self.optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = self.mixup_data(image, label)
        outputs = self.predict(inputs)
        loss = self.mixup_criterion(self.criterion_CE, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        self.optimizer.step()

        return {'loss':loss}
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
class Fishr(ERM):
    def __init__(self, num_classes, cfg):
        super(Fishr, self).__init__(num_classes, cfg)
        
        self.num_groups = cfg.data.FISHR.NUM_GROUPS

        self.network = models.get_net(cfg)
        self.classifier = extend(
            models.get_classifier(self.network._out_features, cfg)
        )
        self.optimizer = None
        
        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(cfg.data.FISHR.EMA, oneminusema_correction=True)
            for _ in range(self.num_groups)
        ]  
        self._init_optimizer()
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            list(self.network.parameters()) + list(self.classifier.parameters()),
            lr = self.cfg.data.LEARNING_RATE,
            momentum = self.cfg.data.MOMENTUM,
            weight_decay = self.cfg.data.WEIGHT_DECAY,
            nesterov=True)
        
    def update(self, minibatch):
        image, mask, label, domain, img_path = minibatch
        #self.network.train()

        all_x = image
        all_y = label
        
        len_minibatches = [image.shape[0]]
        
        all_z = self.network(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.cfg.FISHR.PENALTY_ANNEAL_ITERS:
            penalty_weight = self.cfg.FISHR.LAMBDA
            if self.update_count == self.cfg.FISHR.PENALTY_ANNEAL_ITERS != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True)
            #torch.autograd.grad(outputs=loss,inputs=list(self.classifier.parameters()),retain_graph=True, create_graph=True)
            
        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                #(name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                (name, weights.grad.clone().view(weights.grad.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_groups)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                #print(domain_id)
                #print(bsize)
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_groups):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_groups)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_groups):
            penalty += self.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_groups

    def l2_between_dicts(self, dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        ).pow(2).mean()

# DRGen is built based on Fishr method

class DRGen(Algorithm):
    '''
    Refer to the paper 'DRGen: Domain Generalization in Diabetic Retinopathy Classification' 
    https://link.springer.com/chapter/10.1007/978-3-031-16434-7_61
    
    '''
    def __init__(self, num_classes, cfg):
        super(DRGen, self).__init__(num_classes, cfg)
        algorithm_class = get_algorithm_class('Fishr')
        self.algorithm = algorithm_class(cfg.num_classes, cfg)
        self.optimizer = self.algorithm.optimizer
        
        self.swad_algorithm = AveragedModel(self.algorithm)
        self.swad_algorithm.cuda()
        #swad_cls = getattr(swad_module, 'LossValley')
        #swad_cls = LossValley()
        self.swad = LossValley(None, cfg.data.DRGEN.N_CONVERGENCE, cfg.data.DRGEN.N_TOLERANCE, cfg.data.DRGEN.TOLERANCE_RATIO)
        
    def update(self, minibatch):
        loss_dict_iter = self.algorithm.update(minibatch)
        if self.swad:
            self.swad_algorithm.update_parameters(self.algorithm, step = self.epoch)
        return loss_dict_iter
    
    def validate(self, val_loader, test_loader, writer):
        swad_val_auc = -1
        swad_auc = -1
        if self.epoch <= self.cfg.data.EPOCHS:
            val_auc, val_loss = algorithm_validate(self.algorithm, val_loader, writer, self.epoch, 'val(Fishr)')
            test_auc, test_loss = algorithm_validate(self.algorithm, test_loader, writer, self.epoch, 'test(Fishr)')

            if self.swad:
                def prt_results_fn(results):
                    print(results)

                self.swad.update_and_evaluate(
                    self.swad_algorithm, val_auc, val_loss, prt_results_fn
                )
                
                if self.epoch != self.cfg.data.EPOCHS:
                    self.swad_algorithm = self.swad.get_final_model()
                    swad_val_auc, swad_val_loss = algorithm_validate(self.swad_algorithm, val_loader, writer, self.epoch, 'val')
                    swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.epoch, 'test')
                    
                    if hasattr(self.swad, "dead_valley") and self.swad.dead_valley:
                        logging.info("SWAD valley is dead -> not stop !")
                        #break
                    
                    self.swad_algorithm = AveragedModel(self.algorithm)  # reset
            
            if self.epoch == self.cfg.data.EPOCHS:
                self.epoch += 1
                
        else:
            self.swad_algorithm = self.swad.get_final_model()
            logging.warning("Evaluate SWAD ...")
            swad_auc, swad_loss = algorithm_validate(self.swad_algorithm, test_loader, writer, self.cfg.data.EPOCHS + self.cfg.data.VAL_EPOCH , 'test')
            logging.info('(last) swad test auc: {}  loss: {}'.format(swad_auc,swad_loss))
            
        return swad_val_auc, swad_auc    
        
    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.algorithm.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.algorithm.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))
    
    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.algorithm.network.load_state_dict(torch.load(net_path))
        self.algorithm.classifier.load_state_dict(torch.load(classifier_path))
    
    def predict(self, x):
        return self.swad_algorithm.predict(x)
