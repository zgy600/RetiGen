import logging
import algorithms
import os
import time
import re
import glob
import random
import fundusaug as FundusAug
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utilsGDRBench.validate import *
from utilsGDRBench.args import *
from utilsGDRBench.misc import *
import wandb
import pickle
from collections import Counter
import os.path as osp
from torchvision import transforms
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from PIL import Image
from classifier import Classifier, Classifier_RetiGenBench, OriginModel, Classifier_RetiGenBench_GREEN
from image_list import RetiGenBench, RetiGenBench_pseudo_blc
from moco.builder import AdaMoCo
from moco.loader import NCropsTransform
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_distances,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    save_checkpoint,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
)


@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, args, mode="default"):
    multi_view = True
    wandb_dict = dict()
    model.eval()
    logits, gt_labels, indices, output, features = [], [], [], [], []
    logits_mva, output_mva, features_mva = [], [], []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for imgs, labels, idxs, paths in iterator:
        imgs = imgs.to("cuda")
        softmax = torch.nn.Softmax(dim=1)
        feats, logits_cls = model(imgs, cls_only=True)
        output_sf = softmax(logits_cls)
        
        feats_mva = feats.clone()
        logits_cls_mva = logits_cls.clone()
        output_sf_mva = output_sf.clone()
        
        if mode == "blc" :
            if multi_view:
                if args.algorithm != 'GDRNet':
                    _, test_ts, _ = get_transform(args)
                else:
                    _, test_ts, _ = get_pre_FundusAug(args)
                for i, path in enumerate(paths):
                    mv_list = []
                    if 'MFIDDR_4view' in path or 'DRTiD_2view' in path :
                        # print(path)
                        directory, filename = os.path.split(path)
                        name, extension = os.path.splitext(filename)
                        match = re.match(r'(.+)_\d+', name)
                        if match:
                            patient = match.group(1)
                        search_pattern = os.path.join(directory, patient + '*' + str(extension))
                        matching_files = glob.glob(search_pattern)
                        mv_list.append(matching_files)
                        mv_img_path = [item for sublist in mv_list for item in sublist]
                        mv_img = [Image.open(file_path).convert("RGB") for file_path in mv_img_path]
                        mv_img_trans = [test_ts(img) for img in mv_img]
                        mv_img_tensor = torch.stack(mv_img_trans, dim=0).cuda()
                        feats_mv, logits_cls_mv = model(mv_img_tensor, cls_only=True)
                        output_mv = softmax(logits_cls_mv)
                        output_mv_mean = torch.mean(output_mv, dim=0, keepdim=True)
                        feats_mv_mean = torch.mean(feats_mv, dim=0, keepdim=True)
                        logits_cls_mean = torch.mean(logits_cls_mv, dim=0, keepdim=True)
                    
                    output_sf_mva[i, :] = output_mv_mean
                    feats_mva[i, :] = feats_mv_mean
                    logits_cls_mva[i, :] = logits_cls_mean
        
        output.append(output_sf)
        features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)
        
        if mode == "blc" :
            output_mva.append(output_sf_mva)
            features_mva.append(feats_mva)
            logits_mva.append(logits_cls_mva)
    
    features = torch.cat(features)
    logits = torch.cat(logits)
    output = torch.cat(output)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")
    if mode == "blc" :
        features_mva,  logits_mva, output_mva = torch.cat(features_mva), torch.cat(logits_mva), torch.cat(output_mva)

    if args.distributed:
        # features, logits, output, gt_labels, indices = [
        #     concat_all_gather(x) for x in [features, logits, output, gt_labels, indices]
        # ]
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        output = concat_all_gather(output)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)
        ranks = len(dataloader.dataset) % dist.get_world_size()
        features = remove_wrap_arounds(features, ranks)
        logits = remove_wrap_arounds(logits, ranks)
        output = remove_wrap_arounds(output, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)
        # for x in [features, logits, output, gt_labels, indices]:
        #     remove_wrap_arounds(x, len(dataloader.dataset) % dist.get_world_size())
        if mode == "blc" :
            print("blc--------mode")
            features_mva, logits_mva, output_mva = [
            concat_all_gather(x) for x in [features_mva, logits_mva, output_mva]
            ]
            for x in [features_mva, logits_mva, output_mva]:
                remove_wrap_arounds(x, len(dataloader.dataset) % dist.get_world_size())
            
    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    
    label, pred, output = gt_labels.cpu().data.numpy().tolist(), pred_labels.cpu().data.numpy().tolist(), output.cpu().data.numpy().tolist()
    f1 = f1_score(label, pred, average='macro')
    if mode == "auc":
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        logging.info(f"AUC: {auc_ovo:.6f}")
        wandb_dict.update({"Test AUC": auc_ovo})
    if mode == "blc" :
        pred_labels_mva = logits_mva.argmax(dim=1)
        accuracy_mva = (pred_labels_mva == gt_labels).float().mean()
        pred_mva = pred_labels_mva.cpu().data.numpy().tolist()
        output_mva = output_mva.cpu().data.numpy().tolist()
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        f1_mva = f1_score(label, pred_mva, average='macro')
        auc_ovo_mva = roc_auc_score(label, output_mva, average='macro', multi_class='ovo')
        logging.info(f"Accuracy of direct prediction: {accuracy:.6f}")
        logging.info(f"F1 score: {f1:.6f}")
        logging.info(f"AUC: {auc_ovo:.6f}")
    
        logging.info(f"Accuracy multi-view of direct prediction: {accuracy_mva:.10f}")
        logging.info(f"F1 score mva: {f1_mva:.10f}")
        logging.info(f"AUC mva: {auc_ovo_mva:.10f}")
        source_domains = args.data.source_domains
        result_string = f"AUC: {auc_ovo_mva:.10f}\n"
        result_string += f"ACC: {accuracy_mva:.10f}\n"
        result_string += f"F1: {f1_mva:.10f}\n"
        algorithm_name = args.model_src.algorithm
        alpha = args.learn.alpha
        eta = args.learn.eta
        directory = "/media/raid/gongyu/projects/MVDRG/Result_20240430/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = f"/media/raid/gongyu/projects/MVDRG/Result_20240430/_source_{source_domains[0]}_{args.name}.txt"
        # file_path = f"/media/raid/gongyu/projects/MVDRG/Result_RetiGen_GDRNet/_source_a_{alpha}_e_{eta}_.txt"
        print(file_path)
        with open(file_path, 'w') as file:
            file.write(result_string)
    
        wandb_dict["Test Acc"] = accuracy
        wandb_dict["Test F1 score"] = f1
        wandb_dict["Test AUC"] = auc_ovo
    
        wandb_dict["Test Acc mva"] = accuracy_mva
        wandb_dict["Test F1 score mva"] = f1_mva
        wandb_dict["Test AUC mva"] = auc_ovo_mva
    

        probs = F.softmax(logits, dim=1)
        rand_idxs = torch.randperm(len(features)).cuda()
        banks = {
            "features": features[rand_idxs][: args.learn.queue_size],
            "probs": probs[rand_idxs][: args.learn.queue_size],
            "ptr": 0,
        }

        # refine predicted labels
        pred_labels, probs_new, acc = refine_predictions(
            features, probs, banks, args=args, gt_labels=gt_labels
        )
        wandb_dict["Test Post Acc"] = acc
    
        ########### multi-view refine
        probs_mva = F.softmax(logits_mva, dim=1)
        rand_idxs_mva = torch.randperm(len(features_mva)).cuda()
        banks_mva = {
            "features": features_mva[rand_idxs_mva][: args.learn.queue_size],
            "probs": probs_mva[rand_idxs_mva][: args.learn.queue_size],
            "ptr": 0,
        }

        # refine predicted labels
        pred_labels_mva_refine, probs_mva_refine, acc_mva_refine = refine_predictions(
            features_mva, probs_mva, banks_mva, args=args, gt_labels=gt_labels
        )
        wandb_dict["Test Post Acc mva refine"] = acc_mva_refine

        accuracy_mva_refine = (pred_labels_mva_refine == gt_labels).float().mean() * 100
        accuracy = (pred_labels == gt_labels).float().mean() * 100
    
        label = gt_labels.cpu().data.numpy().tolist()
        pred = pred_labels.cpu().data.numpy().tolist()
        probs_new = probs_new.cpu().data.numpy().tolist()
        f1 = f1_score(label, pred, average='macro')
        auc_ovo = roc_auc_score(label, probs_new, average='macro', multi_class='ovo')
        wandb_dict["Test Acc refined"] = accuracy
        wandb_dict["Test F1 score refined"] = f1
        wandb_dict["Test AUC refined"] = auc_ovo
    
        pred_mva_refine = pred_labels_mva_refine.cpu().data.numpy().tolist()
        probs_mva_refine = probs_mva_refine.cpu().data.numpy().tolist()
        f1_mva_refine = f1_score(label, pred_mva_refine, average='macro')
        auc_ovo_mva_refine = roc_auc_score(label, probs_mva_refine, average='macro', multi_class='ovo')
        wandb_dict["Test Acc mva refined"] = accuracy_mva_refine
        wandb_dict["Test F1 score mva refined"] = f1_mva_refine
        wandb_dict["Test AUC mva refined"] = auc_ovo_mva_refine
    
        pseudo_item_list = []
        for pred_label, idx in zip(pred_labels, indices):
            img_path, _, img_file = dataloader.dataset.item_list[idx]
            pseudo_item_list.append((img_path, int(pred_label), img_file))
        logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

        if use_wandb(args):
            wandb.log(wandb_dict)

        return pseudo_item_list, banks
        
    wandb_dict.update({"Test Acc": accuracy, "Test F1 score": f1})
    if use_wandb(args):
        wandb.log(wandb_dict)
    
    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: args.learn.queue_size],
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
    }
    pred_labels, probs_new, acc = refine_predictions(features, probs, banks, args=args, gt_labels=gt_labels)
    wandb_dict["Test Post Acc"] = acc
    if mode == "default":
        pseudo_item_list = []
        for pred_label, idx in zip(pred_labels, indices):
            img_path, _, img_file = dataloader.dataset.item_list[idx]
            pseudo_item_list.append((img_path, int(pred_label), img_file))
        logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")
        return pseudo_item_list, banks

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, args):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, args.learn.dist_type)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.learn.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

@torch.no_grad()
def update_labels(banks, idxs, features, logits, args):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points
    if args.distributed:
        idxs = concat_all_gather(idxs)
        features = concat_all_gather(features)
        logits = concat_all_gather(logits)

    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy

@torch.no_grad()
def refine_predictions_multi_view(
    features,
    probs,
    banks,
    args,
    gt_labels=None,
):
    if args.learn.refine_method == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, args
        )
    elif args.learn.refine_method is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{args.learn.refine_method} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy

def get_augmentation_versions(args):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in args.learn.aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(args.data.aug_type))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == "t":
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform

def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": extra_params,
                    "lr": args.optim.lr * 10,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer

def train_target_domain_RetiGenBench(args):
    print("train_target_domain_RetiGenBench")
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )
    if args.algorithm != 'GDRNet':
        _, test_ts, _ = get_transform(args)
    else:
        _, test_ts, _ = get_pre_FundusAug(args)
    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        root = osp.abspath(osp.expanduser(args.data.data_root))
        print('##############################')
        print(root)
        dummy_dataset = RetiGenBench(root = args.data.data_root, source_domains= args.data.source_domains, target_domains = args.data.target_domains, \
                                mode = 'test', trans_basic=test_ts)
        
        data_length = len(dummy_dataset)             
        args.learn.queue_size = data_length
        del dummy_dataset

    # src_model_path = os.path.join('./result/fundusaug', args.output)
    src_model_path = args.output
    net_path = os.path.join(src_model_path, 'best_model.pth')
    classifier_path = os.path.join(src_model_path, 'best_classifier.pth')
    
    print("###############################################")
    if args.model_src.algorithm == 'GREEN':
        print("#GREEN#")
        src_model = Classifier_RetiGenBench_GREEN(args.model_src, net_path, classifier_path)
        momentum_model = Classifier_RetiGenBench_GREEN(args.model_src, net_path, classifier_path)
    else: 
        src_model = Classifier_RetiGenBench(args.model_src, net_path, classifier_path)
        momentum_model = Classifier_RetiGenBench(args.model_src, net_path, classifier_path)
    
    model = AdaMoCo(
        src_model,
        momentum_model,
        K=args.model_tta.queue_size,
        m=args.model_tta.m,
        T_moco=args.model_tta.T_moco,
    ).cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model, device_ids=[args.gpu])
        
        
    logging.info("Generate initial pseudo labels")
    val_transform = get_augmentation("test")
    val_dataset = RetiGenBench(root = args.data.data_root, 
                           source_domains= args.data.source_domains, 
                           target_domains = args.data.target_domains,
                                mode = 'test', trans_basic=test_ts, transform=val_transform)
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, sampler=val_sampler, num_workers=2
    )
    
    # algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    # algorithm = algorithm_class(args.num_classes, args)
    # algorithm.cuda()
    # algorithm.renew_model(src_model_path)
    
    # print(args.algorithm)
    # print(args.data.source_domains)

    # _, test_auc = algorithm.generate_pseudo_labels(val_loader)
    # print('test_auc:{}'.format(test_auc))    
    
    logging.info(f"1 - Created target model")
    val_dataset_pseudo_blc = RetiGenBench_pseudo_blc(root = args.data.data_root, 
                           source_domains= args.data.source_domains, 
                           target_domains = args.data.target_domains,
                                mode = 'pseudo_blc', trans_basic=test_ts, transform=val_transform, Args=args)

    val_sampler_pseudo_blc = (
        DistributedSampler(val_dataset_pseudo_blc, shuffle=False) if args.distributed else None
    )
    
    val_loader_pseudo_blc = DataLoader(
        val_dataset_pseudo_blc, batch_size=256, sampler=val_sampler_pseudo_blc, num_workers=2
    )

    pseudo_item_list, banks = eval_and_label_dataset(
        val_loader_pseudo_blc, model, banks=None, args=args
    )
    
    logging.info("2 - Computed initial pseudo labels")

    # Training data
    train_transform = get_augmentation_versions(args)

    train_dataset_pseudo_blc = RetiGenBench_pseudo_blc(root = args.data.data_root, 
                             source_domains= args.data.source_domains, 
                             target_domains = args.data.target_domains,
                             mode = 'pseudo_blc', trans_basic=test_ts,
                             transform=train_transform, Args=args)
    
    train_sampler_pseudo_blc = DistributedSampler(train_dataset_pseudo_blc) if args.distributed else None
    
    train_loader_pseudo_blc = DataLoader(
        train_dataset_pseudo_blc,
        batch_size=args.data.batch_size,
        shuffle=(train_sampler_pseudo_blc is None),
        num_workers=args.data.workers,
        pin_memory=True,
        sampler=train_sampler_pseudo_blc,
        drop_last=False,
    )

    args.learn.full_progress = args.learn.epochs * len(train_loader_pseudo_blc)
    logging.info("3 - Created train/val loader")

    # define loss function (criterion) and optimizer
    optimizer = get_target_optimizer(model, args)
    logging.info("4 - Created optimizer")
    logging.info("Start training...")
    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        
        # if epoch%2 == 0:
        #     print("evaluating epoch: {}".format(epoch))
        #     eval_and_label_dataset(val_loader, model, banks, args, "auc")
        if args.curve==True:
            eval_and_label_dataset(val_loader, model, banks, args, "auc")
        if args.distributed:
            train_sampler_pseudo_blc.set_epoch(epoch)

        # train for one epoch
        # train_epoch(train_loader, model, banks, optimizer, epoch, args)

        print("--training train_epoch train_loader_pseudo_blc")
        train_epoch(train_loader_pseudo_blc, model, banks, optimizer, epoch, args)

            #####################################################################
        if epoch>=(args.learn.epochs-1):
            print("evaluating epoch: {}".format(epoch))
            eval_and_label_dataset(val_loader, model, banks, args, "blc")

            
    if is_master(args):
        filename = f"checkpoint_{epoch:04d}_{args.data.src_domain}-{args.data.tgt_domain}-{args.sub_memo}_{args.seed}.pth.tar"
        save_path = os.path.join(args.log_dir, "output", filename)
        save_checkpoint(model, optimizer, epoch, save_path=save_path)
        logging.info(f"Saved checkpoint {save_path}")

def train_epoch(train_loader, model, banks, optimizer, epoch, args):
    if args.algorithm != 'GDRNet':
        _, test_ts, _ = get_transform(args)
    else:
        _, test_ts, _ = get_pre_FundusAug(args)
    
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd = AverageMeter("CLS-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    # make sure to switch to train mode
    model.train()

    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    
    for i, data in enumerate(train_loader):
        # unpack and move data
        #########################################
        images, label, idxs, paths = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )

        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)

        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, args=args
            )
            for j, path in enumerate(paths):
                mv_list = []
                if 'MFIDDR_4view' in path or 'DRTiD_2view' in path:
                    if j%10==0:
                        print("---start---multi-view-pseudo-labels")
                    directory, filename = os.path.split(path)
                    name, extension = os.path.splitext(filename)
                    match = re.match(r'(.+)_\d+', name)
                    if match:
                        patient = match.group(1)
                    search_pattern = os.path.join(directory, patient + '*' + str(extension))
                    matching_files = glob.glob(search_pattern)
                    mv_list.append(matching_files)
                    mv_img_path = [item for sublist in mv_list for item in sublist]
                    mv_img = [Image.open(file_path).convert("RGB") for file_path in mv_img_path]
                    mv_img_trans = [test_ts(img) for img in mv_img]
                    mv_img_tensor = torch.stack(mv_img_trans, dim=0).cuda()
                    
                    feats_mv, logits_mv = model(mv_img_tensor, cls_only=True)
                    feats_mv_mean = torch.mean(feats_mv, dim=0, keepdim=True)
                    logits_mv_mean = torch.mean(logits_mv, dim=0, keepdim=True)
                    prob = F.softmax(logits_mv_mean, dim=1)
                    if args.knn_mva_pseudo_labels:
                        prob_mean = (prob * len(logits_mv) + probs_w[j,:] * args.learn.num_neighbors) / (len(logits_mv)+args.learn.num_neighbors)
                    else:
                        prob_mean = (prob * 0 + probs_w[j,:] * args.learn.num_neighbors) / (0+args.learn.num_neighbors)
                    # probs_w[j,:] = prob
                    # _, pred_label = prob.max(dim=1)
                    probs_w[j,:] = prob_mean
                    _, pred_label = prob_mean.max(dim=1)
                    pseudo_labels_w[j] = pred_label
                else:
                    # print("---without-multi-view-pseudo-labels")
                    pass
            # output_list.append(probs_w.cpu().data.numpy())
            # label_list.append(label.cpu().data.numpy())
            # pseudo_label_list.append(pseudo_labels_w.cpu().data.numpy())
            
        _, logits_q, logits_ins, keys = model(images_q, images_k)
        # update key features and corresponding pseudo labels
        model.update_memory(keys, pseudo_labels_w)
        ##################################################################
        # model.update_memory(keys, label)
        # moco instance discrimination
        loss_ins, accuracy_ins = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=model.mem_labels,
            contrast_type=args.learn.contrast_type,
        )
        # instance accuracy shown for only one process to give a rough idea
        top1_ins.update(accuracy_ins.item(), len(logits_ins))

        # classification
        loss_cls, accuracy_psd = classification_loss(
        logits_w, logits_q, pseudo_labels_w, args
            )
        
        top1_psd.update(accuracy_psd.item(), len(logits_w))

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, args)
            if args.learn.eta > 0
            else zero_tensor
        )

        loss = (
            args.learn.alpha * loss_cls
            + args.learn.beta * loss_ins
            + args.learn.eta * loss_div
        )
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            feats_w, logits_w = model.momentum_model(images_w, return_feats=True)

        update_labels(banks, idxs, feats_w, logits_w, args)

        if use_wandb(args):
            wandb_dict = {
                "loss_cls": args.learn.alpha * loss_cls.item(),
                "loss_ins": args.learn.beta * loss_ins.item(),
                "loss_div": args.learn.eta * loss_div.item(),
                "acc_ins": accuracy_ins.item(),
            }

            wandb.log(wandb_dict, commit=(i != len(train_loader) - 1))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.learn.print_freq == 0:
            progress.display(i)

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool) # (32,16385)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())  # (B, K)
    
    loss = F.cross_entropy(logits_ins, labels_ins)  # logits_ins:torch.Size([32, 16385]), labels_ins:torch.Size([32])

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy

def classification_loss(logits_w, logits_s, target_labels, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels, args)
        accuracy = calculate_acc(logits_w, target_labels)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_cls = cross_entropy_loss(logits_s, target_labels, args)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{args.learn.ce_sup_type} CE supervision type not implemented."
        )
    return loss_cls, accuracy

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div

def diversification_loss(logits_w, logits_s, args):
    if args.learn.ce_sup_type == "weak_weak":
        loss_div = div(logits_w)
    elif args.learn.ce_sup_type == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div

def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss

def cross_entropy_loss(logits, labels, args):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")

def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss

def get_transform(cfg):
    
    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        transforms.RandomResizedCrop(re_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        normalize,
    ])
    
    tra_test = transforms.Compose([
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    tra_mask= transforms.Compose([
                transforms.Resize(re_size),
                transforms.ToTensor()])
    
    return tra_train, tra_test, tra_mask

def get_pre_FundusAug(cfg):

    size = 256
    re_size = 224
    normalize = get_normalize()
    tra_train = transforms.Compose([
        transforms.Resize(size),
        transforms.ColorJitter( brightness=cfg.TRANSFORM.COLORJITTER_B, \
                                contrast=cfg.TRANSFORM.COLORJITTER_C, \
                                saturation=cfg.TRANSFORM.COLORJITTER_S, \
                                hue=cfg.TRANSFORM.COLORJITTER_H),
        transforms.ToTensor()
    ])
    
    tra_test = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(re_size),
            transforms.ToTensor(),
            normalize])
    
    tra_mask= transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()])
    
    return tra_train, tra_test, tra_mask

def get_post_FundusAug(cfg):
    aug_prob = cfg.TRANSFORM.AUGPROB
    size = 256
    re_size = 224
    normalize = get_normalize()

    tra_fundus_1 = FundusAug.Compose([
        FundusAug.Sharpness(prob = aug_prob),
        FundusAug.Halo(size, prob=aug_prob),
        FundusAug.Hole(size, prob=aug_prob),
        FundusAug.Spot(size, prob=aug_prob),
        FundusAug.Blur(prob=aug_prob)
    ])
    
    tra_fundus_2 = transforms.Compose([
                transforms.RandomCrop(re_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                normalize])
    
    return {'post_aug1':tra_fundus_1, 'post_aug2':tra_fundus_2}

def get_normalize():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    return normalize