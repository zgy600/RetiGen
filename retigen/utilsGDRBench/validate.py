import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
import numpy as np
import pickle
import torch.nn.functional as F
import os, argparse
import os.path as osp
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from torchvision import transforms
import shutil
from collections import Counter
from sklearn.metrics import confusion_matrix
from utilsGDRBench.data_manager import get_post_FundusAug, get_pre_FundusAug, get_transform
import glob
import re
# validate the algorithm by AUC, accuracy and f1 score on val/test datasets
from utils import (
    concat_all_gather,
    remove_wrap_arounds
)

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):    
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)     
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_test(algorithm, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            image.requires_grad_()
            label = label.cuda().long()

            output = algorithm.predict(image)
            
            
            # print(type(output))
            # print(output)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            index_list.append(index)
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
        
        # save pics
        # diff_indices = [i for i, (l, p) in enumerate(zip(label, pred)) if l != p]
        # diff_paths = [index[i] for i in diff_indices]
        # print(diff_paths)
        # print(len(diff_paths))
        # for i, diff_path in enumerate(diff_paths):
        #     directory, filename = os.path.split(diff_path)
        #     directory_new = directory.replace("images", "images_wrong")
        #     name, extension = os.path.splitext(filename)
        #     name_label = label[diff_indices[i]]
        #     name_pred = pred[diff_indices[i]]
        #     name_new = name + '_label_' + str(name_label) + '_pred_' + str(name_pred)
        #     filename_new = name_new + extension
        #     outdir = os.path.join(directory_new, filename_new)
        #     os.makedirs(directory_new, exist_ok=True)
        #     shutil.copy2(diff_path, outdir)
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        cm = confusion_matrix(label, pred)
        print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, acc, f1, loss

def algorithm_generate_pseudo_labels(algorithm, data_loader, epoch, val_type, cfg):
    if cfg.algorithm != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
    root = cfg.data.data_root
    source_domains= cfg.data.source_domains
    target_domains = cfg.data.target_domains
    split_dir = osp.join(root, "splits")
    algorithm_name = cfg.algorithm
    # file_path = osp.join(split_dir, str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt") 
    if cfg.dg_mode == 'DG':
        file_path = osp.join(split_dir, str(cfg.dg_mode) + "_" + str(target_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
        print(file_path)
    elif cfg.dg_mode == 'ESDG':
        file_path = osp.join(split_dir, str(cfg.dg_mode) + "_" + str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
        print(file_path)
    else:
        print("wrong dg_mode")
    
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []
        with open(file_path, 'w') as file:
            for image, label, domain, index in data_loader:
                image = image.cuda()
                image.requires_grad_()
                label = label.cuda().long()
                path = index
                if cfg.distributed:
                    image = concat_all_gather(image)
                    label = concat_all_gather(label)
                    path = concat_all_gather(path)
                    ranks = len(dataloader.dataset) % dist.get_world_size()
                    image = remove_wrap_arounds(image, ranks)
                    label = remove_wrap_arounds(label, ranks)
                    path = remove_wrap_arounds(path, ranks)
                output = algorithm.predict(image)
                loss += criterion(output, label).item()
                _, pred = torch.max(output, 1)
                output_sf = softmax(output)
                for p, pr in zip(path, pred):
                        file.write(f"{p} {pr}\n")
                # print(pred)
                label_list.append(label.cpu().data.numpy())
                pred_list.append(pred.cpu().data.numpy())
                output_list.append(output_sf.cpu().data.numpy())
                index_list.append(index)
                
        # with open(file_path, 'w') as file:
        #     for image, label, domain, index in data_loader:
        #         label = label.cuda().long()
        #         path = index          
        #         for p, pr in zip(path, label):
        #                 file.write(f"{p} {pr}\n")
        #         # print(label)

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
 
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        cm = confusion_matrix(label, pred)
        print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, acc, f1, loss

def algorithm_generate_pseudo_labels_MVLCE(algorithm, data_loader, epoch, val_type, cfg):
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
    root = cfg.DATASET.ROOT
    source_domains= cfg.DATASET.SOURCE_DOMAINS
    target_domains = cfg.DATASET.TARGET_DOMAINS
    split_dir = osp.join(root, "splits")
    algorithm_name = cfg.ALGORITHM
    # file_path = osp.join(split_dir, str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt") 
    if cfg.DG_MODE == 'DG':
        file_path = osp.join(split_dir, str(target_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
        print(file_path)
    elif cfg.DG_MODE == 'ESDG':
        file_path = osp.join(split_dir, str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
        print(file_path)
    else:
        print("wrong dg_mode")
    
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []
        with open(file_path, 'w') as file:
            for image, label, domain, index in data_loader:
                image = image.cuda()
                image.requires_grad_()
                label = label.cuda().long()
                paths = index
                output = algorithm.predict(image)
                for i, path in enumerate(index):
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
                        output_mv = algorithm.predict(mv_img_tensor)
                        output_mv_mean = torch.mean(output_mv, dim=0, keepdim=True)
                        output[i, :] = output_mv_mean
                
                loss += criterion(output, label).item()
                _, pred = torch.max(output, 1)
                output_sf = softmax(output)
                for p, pr in zip(paths, pred):
                        file.write(f"{p} {pr}\n")
                # print(pred)
                label_list.append(label.cpu().data.numpy())
                pred_list.append(pred.cpu().data.numpy())
                output_list.append(output_sf.cpu().data.numpy())
                index_list.append(index)
                

        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
 
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        cm = confusion_matrix(label, pred)
        print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, acc, f1, loss

def algorithm_validate_ttach_test(algorithm, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            image.requires_grad_()
            label = label.cuda().long()

            output = algorithm.predict_tta(image)
            # print(type(output))
            # print(output)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            index_list.append(index)
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
        
        # save pics
        # diff_indices = [i for i, (l, p) in enumerate(zip(label, pred)) if l != p]
        # diff_paths = [index[i] for i in diff_indices]
        # print(diff_paths)
        # print(len(diff_paths))
        # for i, diff_path in enumerate(diff_paths):
        #     directory, filename = os.path.split(diff_path)
        #     directory_new = directory.replace("images", "images_wrong")
        #     name, extension = os.path.splitext(filename)
        #     name_label = label[diff_indices[i]]
        #     name_pred = pred[diff_indices[i]]
        #     name_new = name + '_label_' + str(name_label) + '_pred_' + str(name_pred)
        #     filename_new = name_new + extension
        #     outdir = os.path.join(directory_new, filename_new)
        #     os.makedirs(directory_new, exist_ok=True)
        #     shutil.copy2(diff_path, outdir)
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_ttamv_test(algorithm, data_loader, epoch, val_type, cfg):
    if cfg.ALGORITHM != 'GDRNet':
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
        
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            image.requires_grad_()
            label = label.cuda().long()
            output = algorithm.predict(image)
            
            for i, path in enumerate(index):
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
                    output_mv = algorithm.predict(mv_img_tensor)
                    output_mv_mean = torch.mean(output_mv, dim=0, keepdim=True)
                    output[i, :] = output_mv_mean
            
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            index_list.append(index)
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
        
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        # cm = confusion_matrix(label, pred)
        # print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        loss = loss / len(data_loader)

    algorithm.train()
    return auc_ovo, acc, f1, loss

def algorithm_validate_tent_test(algorithm, data_loader, epoch, val_type, cfg):
    if cfg.ALGORITHM != 'GDRNet' :
        train_ts, test_ts, tra_fundus = get_transform(cfg)
    else:
        train_ts, test_ts, tra_fundus = get_pre_FundusAug(cfg)
    log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    loss = 0
    label_list = []
    output_list = []
    pred_list = []
    diff_list = []
    index_list = []
        
    for image, label, domain, index in data_loader:
        image = image.cuda()
        image.requires_grad_()
        label = label.cuda().long()
        
        output = algorithm.predict_tent(image)
        # output = model(image)
        loss += criterion(output, label).item()

        _, pred = torch.max(output, 1)
        output_sf = softmax(output)
            
        label_list.append(label.cpu().data.numpy())
        pred_list.append(pred.cpu().data.numpy())
        output_list.append(output_sf.cpu().data.numpy())
        index_list.append(index)
        
    label = [item for sublist in label_list for item in sublist]
    pred = [item for sublist in pred_list for item in sublist]
    output = [item for sublist in output_list for item in sublist]      
    index = [item for sublist in index_list for item in sublist]
        
        
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average='macro')
    # cm = confusion_matrix(label, pred)
    # print(cm)
    auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
    loss = loss / len(data_loader)

    algorithm.train()
    return auc_ovo, acc, f1, loss

def algorithm_validate_Tau_normalized_test(algorithm, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            image.requires_grad_()
            label = label.cuda().long()

            output = algorithm.predict_tta_Tau_normalized(image)
            
            # print(type(output))
            # print(output)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            index_list.append(index)
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
        
        # save pics
        # diff_indices = [i for i, (l, p) in enumerate(zip(label, pred)) if l != p]
        # diff_paths = [index[i] for i in diff_indices]
        # print(diff_paths)
        # print(len(diff_paths))
        # for i, diff_path in enumerate(diff_paths):
        #     directory, filename = os.path.split(diff_path)
        #     directory_new = directory.replace("images", "images_wrong")
        #     name, extension = os.path.splitext(filename)
        #     name_label = label[diff_indices[i]]
        #     name_pred = pred[diff_indices[i]]
        #     name_new = name + '_label_' + str(name_label) + '_pred_' + str(name_pred)
        #     filename_new = name_new + extension
        #     outdir = os.path.join(directory_new, filename_new)
        #     os.makedirs(directory_new, exist_ok=True)
        #     shutil.copy2(diff_path, outdir)
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        cm = confusion_matrix(label, pred)
        print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        print('test_auc:{}'.format(auc_ovo))
        print('test_acc:{}'.format(acc))
        print('test_f1:{}'.format(f1))
        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def logit_adjustment_test(algorithm, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        diff_list = []
        index_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda()
            image.requires_grad_()
            label = label.cuda().long()

            output = algorithm.predict(image)
            numbers = torch.tensor([3808, 1085, 2899, 931, 587], dtype=torch.float)
            total_sum = torch.sum(numbers)
            probabilities = numbers / total_sum
            probabilities = probabilities.cuda()
            output1 = output
            output2 = output - 0.5 * torch.log(probabilities)
            alpha = 0
            output = alpha * output1 + (1 - alpha) * output2
            # print(type(output))
            # print(output)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            index_list.append(index)
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]      
        index = [item for sublist in index_list for item in sublist]
        
        # save pics
        # diff_indices = [i for i, (l, p) in enumerate(zip(label, pred)) if l != p]
        # diff_paths = [index[i] for i in diff_indices]
        # print(diff_paths)
        # print(len(diff_paths))
        # for i, diff_path in enumerate(diff_paths):
        #     directory, filename = os.path.split(diff_path)
        #     directory_new = directory.replace("images", "images_wrong")
        #     name, extension = os.path.splitext(filename)
        #     name_label = label[diff_indices[i]]
        #     name_pred = pred[diff_indices[i]]
        #     name_new = name + '_label_' + str(name_label) + '_pred_' + str(name_pred)
        #     filename_new = name_new + extension
        #     outdir = os.path.join(directory_new, filename_new)
        #     os.makedirs(directory_new, exist_ok=True)
        #     shutil.copy2(diff_path, outdir)
        
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        cm = confusion_matrix(label, pred)
        print(cm)
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        print('test_acc:{}'.format(acc))
        print('test_f1:{}'.format(f1))
        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_test_saliency_map(algorithm, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    softmax = torch.nn.Softmax(dim=1)
    loss = 0
    label_list = []
    output_list = []
    pred_list = []
    index_list = []

    for image, label, domain, index in data_loader:
        image = image.cuda()
        # image.requires_grad_()
        label = label.cuda().long()
        batch = image.size(0)
        output = algorithm.predict(image)
        loss += criterion(output, label).item()

        _, pred = torch.max(output, 1)
        output_sf = softmax(output)
        
        # Catch the output
        output_idx = output.argmax(1)
        label_list.append(label.cpu().data.numpy())
        pred_list.append(pred.cpu().data.numpy())
        output_list.append(output_sf.cpu().data.numpy())
        index_list.append(index)

        for i in range(batch):
            img = image[i]
            
            # Reshape the image (because the model use 
            # 4-dimensional tensor (batch_size, channel, width, height))
            img = img.reshape(1, 3, 224, 224)

            # Set the requires_grad_ to the image for retrieving gradients
            img.requires_grad_()

            # Retrieve output from the image
            output_saliency = algorithm.predict(img)

            # Catch the output
            output_idx = output_saliency.argmax()
            output_max = output_saliency[0, output_idx]

            # Do backpropagation to get the derivative of the output based on the image
            output_max.backward()

            # Retireve the saliency map and also pick the maximum value from channels on each pixel.
            # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
            
            saliency, _ = torch.max(img.grad.data.abs(), dim=1) 
            saliency = saliency.reshape(224, 224)

            
            # save saliency map
            img_path = index[i]
            directory, filename = os.path.split(img_path)
            name, extension = os.path.splitext(filename)
            directory_new = directory.replace("images", "images_saliency_map") 
            name_new = name + '_label_' + str(label[i].item()) + '_pred_' + str(pred[i].item()) + '_saliency_map'
            filename_new = name_new + extension
            # outdir = os.path.join(directory_new, filename_new)
            x_sample = Image.fromarray((saliency.cpu().numpy() * 255).astype(np.uint8)) 
            os.makedirs(directory_new, exist_ok=True)
            x_sample.save(os.path.join(directory_new, filename_new))
            
            target_path = os.path.join(directory_new, filename)
            # 使用 shutil.copy2() 复制文件
            shutil.copy2(img_path, target_path)
            

        
    label = [item for sublist in label_list for item in sublist]
    pred = [item for sublist in pred_list for item in sublist]
    output = [item for sublist in output_list for item in sublist]      
    index = [item for sublist in index_list for item in sublist]
        
    diff_indices = [i for i, (l, p) in enumerate(zip(label, pred)) if l != p]
    diff_paths = [index[i] for i in diff_indices]
    # print(diff_paths)
    # print(len(diff_paths))
    for i, diff_path in enumerate(diff_paths):
        directory, filename = os.path.split(diff_path)
        directory_new = directory.replace("images", "images_wrong")
        name, extension = os.path.splitext(filename)
        name_label = label[diff_indices[i]]
        name_pred = pred[diff_indices[i]]
        name_new = name + '_label_' + str(name_label) + '_pred_' + str(name_pred)
        filename_new = name_new + extension
        outdir = os.path.join(directory_new, filename_new)
        os.makedirs(directory_new, exist_ok=True)
        shutil.copy2(diff_path, outdir)
        
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average='macro')
    
    auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

    loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_tta_test(algorithm, training_domains, data_loader, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    target_size = (224, 224)
    N = len(training_domains)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    average_values_dict = {
    'APTOS': {'mean': [0.2841766, -0.23802036, -0.44554743, 0.8143348],
              'std': [0.53095317, 0.6137561, 0.4887799, 0.3362168]},
    'MFIDDR_1view': {'mean': [0.43545985, -0.6062746, -0.01307655, 1.0771058],
                     'std': [0.73270774, 0.83608466, 0.7059133, 0.38208315]},
    'IDRID': {'mean': [0.2806504, 0.04437273, -0.75075656, 0.63489187],
              'std': [0.45809016, 0.5007796, 0.4153973, 0.32766306]},
    'RLDR': {'mean': [0.2819445, -0.46733987, -0.17754468, 0.5210681],
             'std': [0.65641, 0.820638, 0.6075364, 0.43660662]},
    'DEEPDR': {'mean': [0.40794414, -0.20940323, -0.43452525, 0.53553116],
               'std': [0.58025515, 0.6467379, 0.49657184, 0.37210035]},
    'DRTiD_1view': {'mean': [0.672681, -0.52939606, -0.31786436, 0.77008563],
                    'std': [0.52052486, 0.6000187, 0.49166602, 0.33825636]},
    'MFIDDR_4viewss': {'mean': [0.4880714, -0.5158253, 0.01185736, 1.0561396],
                       'std': [0.74223924, 0.84591156, 0.72839785, 0.37787625]},
    }
    # average_values_list = torch.Tensor([average_values_dict[domain] for domain in training_domains])
    
    new_average_values_dict = {domain: average_values_dict[domain] for domain in training_domains}
    
    # print(new_average_values_dict)
    all_means = []
    all_stds = []

    # get VAE (first-stage model)
    vae_path = 'v2-inference-v-first-stage-VAE.yaml'
    vae_config = OmegaConf.load(f"{vae_path}")
    vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))

    pl_sd = torch.load("checkpoints/768-v-ema-first-stage-VAE.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    vae_model.load_state_dict(sd, strict=True)

    vae_model.eval()
    vae_model = vae_model.to(device)
    scale_factor = vae_config.first_stage_config.scale_factor
    
    # 遍历每个数据集的平均值并累加
    for dataset_values in new_average_values_dict.values():
        all_means.append(dataset_values['mean'])
        all_stds.append(dataset_values['std'])
    
    type(all_means)
    # 计算所有数据集的平均值
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.mean(all_stds, axis=0)
    overall_mean_tensor = torch.tensor(overall_mean, dtype=torch.float32).view(1, -1)
    overall_std_tensor = torch.tensor(overall_std, dtype=torch.float32).view(1, -1)

    overall_mean_std = torch.cat((overall_mean_tensor, overall_std_tensor), dim=1)
    domain_mean_std = torch.cat((torch.tensor(all_means, dtype=torch.float64), torch.tensor(all_stds, dtype=torch.float64)), dim=1)
    # print(type(domain_mean_std))
    # print(overall_mean)
    # print("Overall Mean across all datasets:", overall_mean)
    # print("Overall Std across all datasets:", overall_std)
    # print(overall_mean_std)
    # 计算threshold
    a = 3
    gamma = 0.2 # output fusion
    Lambda = 0.8 # style shift
    print(gamma)
    print(Lambda)
    # print(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = a/N*torch.sum(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    
    
    with open('mean-std.pkl', 'rb') as file:
        mean_std = pickle.load(file) # dict list
    
    # source domain mean-std
    filtered_mean_std = [
    result_dict for result_dict in mean_std if result_dict['dataset_name'] in training_domains]
    
    # print(len(training_domains))
    
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, index in data_loader:
            # print(index)
            # print(type(index))
            image = image.cuda() # (16, 3, 224, 224)
            label = label.cuda().long() # torch.Tensor
            image_replace = image.clone()
            # print(type(label))
            
            img_mean_std = [
                next(result_dict for result_dict in mean_std if result_dict['path'] == path)
                for path in index]
            
            first_tensor_shape = torch.cat((torch.tensor(img_mean_std[0]['mean']), torch.tensor(img_mean_std[0]['std'])), dim=0).view(1, -1).shape
            content_tensor = torch.empty(0, first_tensor_shape[1])
            for result_dict in img_mean_std:
                mean__std = torch.cat((torch.tensor(result_dict['mean'], dtype=torch.float64), torch.tensor(result_dict['std'], dtype=torch.float64)), dim=0).view(1, -1)
                content_tensor = torch.cat((content_tensor, mean__std), dim=0)
            # print(content_tensor.shape) # (16, 8)
            # print(content_tensor)

            replace_indices = []
            for i, img in enumerate(img_mean_std):
                content_mean_std = torch.cat((torch.tensor(img['mean'], dtype=torch.float64), torch.tensor(img['std'], dtype=torch.float64)), dim=0).view(1, -1)
                euclid = 1/N*torch.sum(torch.norm(content_mean_std - domain_mean_std, p=2, dim=1))
                # print(euclid)
                # replace_indices.append(1 if euclid > threshold else 0)
                # replace image
                if euclid > threshold:
                    init_img_content = img['path']
                    init_seg_content = init_img_content.replace("images", "masks")
                    min_l2 = float('inf')
                    max_l2 = float('-inf')
                    max_cos_sim = float('-inf')
                    min_l2_path = None
                    max_l2_path = None
                    max_cos_sim_path = None
                    for j, style_info in enumerate(filtered_mean_std):
                        style_tensor = torch.cat((torch.tensor(style_info['mean'], dtype=torch.float64), 
                                                  torch.tensor(style_info['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        l2 = torch.sum(torch.norm(style_tensor - content_mean_std, p=2, dim=1))
                        cos_sim = F.cosine_similarity(style_tensor, content_mean_std, dim=1)
                        # 更新最小值和对应索引
                        if l2 < min_l2:
                            min_l2 = l2
                            min_l2_path = style_info['path']
                        if l2 > max_l2:
                            max_l2 = l2
                            max_l2_path = style_info['path']
                        if cos_sim > max_cos_sim:
                            max_cos_sim = cos_sim
                            max_cos_sim_path = style_info['path']
                    # print(min_l2_path)
                    # print(max_cos_sim_path)
                    
                
                    init_img_style = min_l2_path
                    init_seg_style = min_l2_path.replace("images", "masks")
                    
                    # init_img_style = max_l2_path
                    # init_seg_style = max_l2_path.replace("images", "masks")
                
                    assert os.path.isfile(init_img_content)
                    init_image = load_img(init_img_content).to(device)
                    init_latent_content = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                    # print('latent vector shape (content img): {}'.format(init_latent_content.shape))

                    assert os.path.isfile(init_img_style)
                    init_image = load_img(init_img_style).to(device)
                    init_latent_style = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                    # print('latent vector shape (style img): {}'.format(init_latent_style.shape))

                    assert os.path.isfile(init_seg_content)
                    init_content_seg = load_img_seg(init_seg_content).to(device)

                    assert os.path.isfile(init_seg_style)
                    init_style_seg = load_img_seg(init_seg_style).to(device)
                
                    x_samples = list()
                    content_latent_size = init_latent_content.shape[2:]
                    style_latent_size   = init_latent_style.shape[2:]
                    # print(content_latent_size)
                    # print(style_latent_size)
                    latent_seg_content = F.interpolate(init_content_seg, size=content_latent_size, mode='nearest')
                    latent_seg_style   = F.interpolate(init_style_seg, size=style_latent_size, mode='nearest')

                    # new_latent = adaptive_instance_normalization_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style)
                    # Lambda style shift parameter
                    new_latent = mixstyle_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style, Lambda)
                    decode_imgs = vae_decode(vae_model, new_latent, scale_factor)
                    x_samples.append(decode_imgs[0])

                    for x_sample in x_samples:
                        resize_transform = transforms.Resize(target_size)
                        resized_img = resize_transform(x_sample)
                        # # print(type(resized_img))
                        # # print(resized_img.size())
                        # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        # img1 = Image.fromarray(x_sample.astype(np.uint8))
                        
                        # #############################################bug
                        # print('origin') # 原始图像
                        # print(img['path']) # 原始图像
                        # print('origin_mask') # 原始mask
                        # print(init_seg_content) # 原始mask
                        # print('style') # 原始图像
                        # print(init_img_style) # 风格图像
                        # print('style_mask') # 风格mask
                        # print(init_seg_style) # 风格图像
                        
                        
                        # original_path = img['path']
                        # directory, filename = os.path.split(original_path)
                        # outdir = directory.replace("images", "imagesreplace")
                        # os.makedirs(outdir, exist_ok=True)
                        # img1.save(os.path.join(outdir,str(filename)))
                        
                        # style_path = init_img_style
                        # directory_style, filename_style = os.path.split(style_path)
                        
                        # # 移动style image
                        # filename_style = filename.replace(".png", "_style.png")
                        # target_path = os.path.join(outdir,str(filename_style))
                        # # 使用 shutil.copy2() 复制文件
                        # shutil.copy2(style_path, target_path)
                        # filename_origin = filename.replace(".png", "_origin.png")
                        # target_path = os.path.join(outdir,str(filename_origin))
                        # shutil.copy2(original_path, target_path)
                        
                
                        # imgg = Image.fromarray(x_sample.astype(np.uint8))
                        # print(x_sample.size())
                        # print(osp.join(outdir, str(file)))
                        # print(type(osp.join(outdir, str(file))))
                    image_replace[i] = resized_img.unsqueeze(0)
                    
            output_origin = algorithm.predict(image)
            output_replace = algorithm.predict(image_replace)
            output = gamma * output_origin + (1 - gamma) * output_replace
            
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_tta_topK_test(algorithm, training_domains, data_loader, epoch, val_type):
    # diffusion tta topK
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    target_size = (224, 224)
    N = len(training_domains)
    K = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    average_values_dict = {
    'APTOS': {'mean': [0.2841766, -0.23802036, -0.44554743, 0.8143348],
              'std': [0.53095317, 0.6137561, 0.4887799, 0.3362168]},
    'MFIDDR_1view': {'mean': [0.43545985, -0.6062746, -0.01307655, 1.0771058],
                     'std': [0.73270774, 0.83608466, 0.7059133, 0.38208315]},
    'IDRID': {'mean': [0.2806504, 0.04437273, -0.75075656, 0.63489187],
              'std': [0.45809016, 0.5007796, 0.4153973, 0.32766306]},
    'RLDR': {'mean': [0.2819445, -0.46733987, -0.17754468, 0.5210681],
             'std': [0.65641, 0.820638, 0.6075364, 0.43660662]},
    'DEEPDR': {'mean': [0.40794414, -0.20940323, -0.43452525, 0.53553116],
               'std': [0.58025515, 0.6467379, 0.49657184, 0.37210035]},
    'DRTiD_1view': {'mean': [0.672681, -0.52939606, -0.31786436, 0.77008563],
                    'std': [0.52052486, 0.6000187, 0.49166602, 0.33825636]},
    'MFIDDR_4viewss': {'mean': [0.4880714, -0.5158253, 0.01185736, 1.0561396],
                       'std': [0.74223924, 0.84591156, 0.72839785, 0.37787625]},
    }
    # average_values_list = torch.Tensor([average_values_dict[domain] for domain in training_domains])
    
    new_average_values_dict = {domain: average_values_dict[domain] for domain in training_domains}
    
    # print(new_average_values_dict)
    all_means = []
    all_stds = []

    # get VAE (first-stage model)
    vae_path = 'v2-inference-v-first-stage-VAE.yaml'
    vae_config = OmegaConf.load(f"{vae_path}")
    vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))

    pl_sd = torch.load("checkpoints/768-v-ema-first-stage-VAE.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    vae_model.load_state_dict(sd, strict=True)

    vae_model.eval()
    vae_model = vae_model.to(device)
    scale_factor = vae_config.first_stage_config.scale_factor
    
    # 遍历每个数据集的平均值并累加
    for dataset_values in new_average_values_dict.values():
        all_means.append(dataset_values['mean'])
        all_stds.append(dataset_values['std'])
    
    type(all_means)
    # 计算所有数据集的平均值
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.mean(all_stds, axis=0)
    overall_mean_tensor = torch.tensor(overall_mean, dtype=torch.float32).view(1, -1)
    overall_std_tensor = torch.tensor(overall_std, dtype=torch.float32).view(1, -1)

    overall_mean_std = torch.cat((overall_mean_tensor, overall_std_tensor), dim=1)
    domain_mean_std = torch.cat((torch.tensor(all_means, dtype=torch.float64), torch.tensor(all_stds, dtype=torch.float64)), dim=1)

    a = 3
    gamma = 0.8 # output fusion
    Lambda = 0.2 # style shift
    print(gamma)
    print(Lambda)
    # print(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = a/N*torch.sum(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    
    
    with open('mean-std.pkl', 'rb') as file:
        mean_std = pickle.load(file) # dict list
    
    # source domain mean-std
    filtered_mean_std = [
    result_dict for result_dict in mean_std if result_dict['dataset_name'] in training_domains]
    
    # print(len(training_domains))
    
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, index in data_loader:
            image = image.cuda() # (16, 3, 224, 224)
            label = label.cuda().long() # torch.Tensor
            image_list = [image.clone() for _ in range(K)]
            # print(type(label))
            
            img_mean_std = [
                next(result_dict for result_dict in mean_std if result_dict['path'] == path)
                for path in index]
            
            first_tensor_shape = torch.cat((torch.tensor(img_mean_std[0]['mean']), torch.tensor(img_mean_std[0]['std'])), dim=0).view(1, -1).shape
            content_tensor = torch.empty(0, first_tensor_shape[1])
            for result_dict in img_mean_std:
                mean__std = torch.cat((torch.tensor(result_dict['mean'], dtype=torch.float64), torch.tensor(result_dict['std'], dtype=torch.float64)), dim=0).view(1, -1)
                content_tensor = torch.cat((content_tensor, mean__std), dim=0)
            # print(content_tensor.shape) # (16, 8)
            # print(content_tensor)

            replace_indices = []
            for i, img in enumerate(img_mean_std):
                content_mean_std = torch.cat((torch.tensor(img['mean'], dtype=torch.float64), torch.tensor(img['std'], dtype=torch.float64)), dim=0).view(1, -1)
                euclid = 1/N*torch.sum(torch.norm(content_mean_std - domain_mean_std, p=2, dim=1))
                # print(euclid)
                # replace_indices.append(1 if euclid > threshold else 0)
                # replace image
                if euclid > threshold:
                    init_img_content = img['path']
                    init_seg_content = init_img_content.replace("images", "masks")
                    min_l2 = float('inf')
                    min_l2_path = None
                    topK_min_l2_paths = []
                    for j, style_info in enumerate(filtered_mean_std):
                        style_tensor = torch.cat((torch.tensor(style_info['mean'], dtype=torch.float64), 
                                                  torch.tensor(style_info['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        l2 = torch.sum(torch.norm(style_tensor - content_mean_std, p=2, dim=1))
                        cos_sim = F.cosine_similarity(style_tensor, content_mean_std, dim=1)
                        # 更新最小值和对应索引
                        if l2 < min_l2:
                            min_l2 = l2
                            min_l2_path = style_info['path']
                            
                        # 如果列表为空或者当前l2小于列表中最大的l2，则更新列表
                        if len(topK_min_l2_paths) < K or l2 < topK_min_l2_paths[-1][0]:
                            topK_min_l2_paths.append((l2, style_info['path']))
                            topK_min_l2_paths = sorted(topK_min_l2_paths, key=lambda x: x[0])[:10]
                        # top_min_l2_paths 现在包含了最小的10个 min_l2 和对应的 min_l2_path

                    topK_image_paths = [path for _, path in topK_min_l2_paths]  
                    
                    for l, image_paths in enumerate(topK_image_paths):
                        init_img_style = image_paths
                        init_seg_style = image_paths.replace("images", "masks")

                        assert os.path.isfile(init_img_content)
                        init_image = load_img(init_img_content).to(device)
                        init_latent_content = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space

                        assert os.path.isfile(init_img_style)
                        init_image = load_img(init_img_style).to(device)
                        init_latent_style = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space

                        assert os.path.isfile(init_seg_content)
                        init_content_seg = load_img_seg(init_seg_content).to(device)

                        assert os.path.isfile(init_seg_style)
                        init_style_seg = load_img_seg(init_seg_style).to(device)
                
                        x_samples = list()
                        content_latent_size = init_latent_content.shape[2:]
                        style_latent_size   = init_latent_style.shape[2:]

                        latent_seg_content = F.interpolate(init_content_seg, size=content_latent_size, mode='nearest')
                        latent_seg_style   = F.interpolate(init_style_seg, size=style_latent_size, mode='nearest')

                        new_latent = mixstyle_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style, Lambda)
                        decode_imgs = vae_decode(vae_model, new_latent, scale_factor)
                        x_samples.append(decode_imgs[0])

                        for x_sample in x_samples:
                            resize_transform = transforms.Resize(target_size)
                            resized_img = resize_transform(x_sample)
                    
                        image_list[l][i] = resized_img.unsqueeze(0)
                    
            all_outputs = []
            all_pred = []
            output_origin = algorithm.predict(image)
            for image_i in image_list:
                output_replace = algorithm.predict(image_i)
                output = gamma * output_origin + (1 - gamma) * output_replace
                _, pred = torch.max(output, 1)
                all_outputs.append(output)
                all_pred.append(pred)
            
            all_outputs_tensor = torch.stack(all_outputs)
            all_pred_tensor = torch.stack(all_pred)

            # 通过 all_pred_tensor 找到每张图片最多的预测结果
            final_predictions = torch.mode(all_pred_tensor, dim=0).values
            matching_predictions = (all_pred_tensor == final_predictions.unsqueeze(0))
            merged_output = all_outputs_tensor * matching_predictions.unsqueeze(2)
            output = torch.mean(merged_output, dim=0)
            
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_tta2_test(algorithm, training_domains, data_loader, epoch, val_type):
    # nearest domain center
    
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    target_size = (224, 224)
    N = len(training_domains)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    average_values_dict = {
    'APTOS': {'mean': [0.2841766, -0.23802036, -0.44554743, 0.8143348],
              'std': [0.53095317, 0.6137561, 0.4887799, 0.3362168]},
    'MFIDDR_1view': {'mean': [0.43545985, -0.6062746, -0.01307655, 1.0771058],
                     'std': [0.73270774, 0.83608466, 0.7059133, 0.38208315]},
    'IDRID': {'mean': [0.2806504, 0.04437273, -0.75075656, 0.63489187],
              'std': [0.45809016, 0.5007796, 0.4153973, 0.32766306]},
    'RLDR': {'mean': [0.2819445, -0.46733987, -0.17754468, 0.5210681],
             'std': [0.65641, 0.820638, 0.6075364, 0.43660662]},
    'DEEPDR': {'mean': [0.40794414, -0.20940323, -0.43452525, 0.53553116],
               'std': [0.58025515, 0.6467379, 0.49657184, 0.37210035]},
    'DRTiD_1view': {'mean': [0.672681, -0.52939606, -0.31786436, 0.77008563],
                    'std': [0.52052486, 0.6000187, 0.49166602, 0.33825636]},
    'MFIDDR_4viewss': {'mean': [0.4880714, -0.5158253, 0.01185736, 1.0561396],
                       'std': [0.74223924, 0.84591156, 0.72839785, 0.37787625]},
    }
    # average_values_list = torch.Tensor([average_values_dict[domain] for domain in training_domains])
    
    new_average_values_dict = {domain: average_values_dict[domain] for domain in training_domains}
    new_average_values_list = list({'key': key, **value} for key, value in new_average_values_dict.items())

    # print(new_average_values_dict)
    all_means = []
    all_stds = []

    # get VAE (first-stage model)
    vae_path = 'v2-inference-v-first-stage-VAE.yaml'
    vae_config = OmegaConf.load(f"{vae_path}")
    vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))

    pl_sd = torch.load("checkpoints/768-v-ema-first-stage-VAE.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    vae_model.load_state_dict(sd, strict=True)

    vae_model.eval()
    vae_model = vae_model.to(device)
    scale_factor = vae_config.first_stage_config.scale_factor
    
    # 遍历每个数据集的平均值并累加
    for dataset_values in new_average_values_dict.values():
        all_means.append(dataset_values['mean'])
        all_stds.append(dataset_values['std'])
    
    type(all_means)
    # 计算所有数据集的平均值
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.mean(all_stds, axis=0)
    overall_mean_tensor = torch.tensor(overall_mean, dtype=torch.float32).view(1, -1)
    overall_std_tensor = torch.tensor(overall_std, dtype=torch.float32).view(1, -1)

    overall_mean_std = torch.cat((overall_mean_tensor, overall_std_tensor), dim=1)
    domain_mean_std = torch.cat((torch.tensor(all_means, dtype=torch.float64), torch.tensor(all_stds, dtype=torch.float64)), dim=1)
    # print(type(domain_mean_std))
    # print(overall_mean)
    # print("Overall Mean across all datasets:", overall_mean)
    # print("Overall Std across all datasets:", overall_std)
    # print(overall_mean_std)
    # 计算threshold
    a = 3
    gamma = 0.2 # output fusion
    Lambda = 0.2 # style shift
    print(gamma)
    print(Lambda)
    # print(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = a/N*torch.sum(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = 0
    
    with open('mean-std.pkl', 'rb') as file:
        mean_std = pickle.load(file) # dict list
    
    # source domain mean-std
    filtered_mean_std = [
    result_dict for result_dict in mean_std if result_dict['dataset_name'] in training_domains]
    
    # print(len(training_domains))
    
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, index in data_loader:
            # print(index)
            # print(type(index))
            image = image.cuda() # (16, 3, 224, 224)
            label = label.cuda().long() # torch.Tensor
            # print(type(label))
            image_replace = image.clone()
            
            
            img_mean_std = [
                next(result_dict for result_dict in mean_std if result_dict['path'] == path)
                for path in index]
            
            first_tensor_shape = torch.cat((torch.tensor(img_mean_std[0]['mean']), torch.tensor(img_mean_std[0]['std'])), dim=0).view(1, -1).shape
            content_tensor = torch.empty(0, first_tensor_shape[1])
            for result_dict in img_mean_std:
                mean__std = torch.cat((torch.tensor(result_dict['mean'], dtype=torch.float64), torch.tensor(result_dict['std'], dtype=torch.float64)), dim=0).view(1, -1)
                content_tensor = torch.cat((content_tensor, mean__std), dim=0)
            # print(content_tensor.shape) # (16, 8)
            # print(content_tensor)

            replace_indices = []
            for i, img in enumerate(img_mean_std):
                content_mean_std = torch.cat((torch.tensor(img['mean'], dtype=torch.float64), torch.tensor(img['std'], dtype=torch.float64)), dim=0).view(1, -1)
                euclid = 1/N*torch.sum(torch.norm(content_mean_std - domain_mean_std, p=2, dim=1))
                # print(euclid)
                # replace_indices.append(1 if euclid > threshold else 0)
                # replace image
                if euclid > threshold:
                    init_img_content = img['path']
                    init_seg_content = init_img_content.replace("images", "masks")
                    min_l2 = float('inf')
                    max_l2 = float('-inf')
                    min_l2_domain = float('inf')
                    max_cos_sim = float('-inf')
                    max_cos_sim_domain = float('-inf')
                    min_l2_path = None
                    max_l2_path = None
                    min_l2_domain_name = None
                    max_cos_sim_path = None
                    max_cos_sim_domain_name = None
                    min_l2_domain_tensor = None
                    # 计算距离最近domain
                    
                    for j, training_domain in enumerate(new_average_values_list):
                        # print(training_domain)
                        # print(type(training_domain))
                        domain_tensor = torch.cat((torch.tensor(training_domain['mean'], dtype=torch.float64), 
                                                  torch.tensor(training_domain['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        l2_domain = torch.sum(torch.norm(domain_tensor - content_mean_std, p=2, dim=1))
                        cos_sim_domain = F.cosine_similarity(domain_tensor, content_mean_std, dim=1)
                        # print(training_domain)
                        if l2_domain < min_l2_domain:
                            min_l2_domain = l2_domain
                            min_l2_domain_name = training_domain['key']
                            min_l2_domain_tensor = domain_tensor
                        if cos_sim_domain > max_cos_sim_domain:
                            max_cos_sim_domain = cos_sim_domain
                            max_cos_sim_domain_name = training_domain['key']
                    
                    filtered2_mean_std = [
                        result_dict for result_dict in filtered_mean_std if result_dict['dataset_name']==min_l2_domain_name ]
                    # print(min_l2_domain_name)
                    # print(len(filtered2_mean_std))
                    # print(min_l2_domain_name)
                    # print(type(min_l2_domain_name))
                    for j, style_info in enumerate(filtered2_mean_std):
                        style_tensor = torch.cat((torch.tensor(style_info['mean'], dtype=torch.float64), 
                                                  torch.tensor(style_info['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        # l2 = torch.sum(torch.norm(style_tensor - content_mean_std, p=2, dim=1))
                        # cos_sim = F.cosine_similarity(style_tensor, content_mean_std, dim=1)
                        l2 = torch.sum(torch.norm(style_tensor - min_l2_domain_tensor, p=2, dim=1))
                        cos_sim = F.cosine_similarity(style_tensor, min_l2_domain_tensor, dim=1)
                        # 更新最小值和对应索引
                        if l2 < min_l2:
                            min_l2 = l2
                            min_l2_path = style_info['path']
                        if l2 > max_l2:
                            max_l2 = l2
                            max_l2_path = style_info['path']
                        if cos_sim > max_cos_sim:
                            max_cos_sim = cos_sim
                            max_cos_sim_path = style_info['path']
                    # print(min_l2_path)
                    # print(max_cos_sim_path)
                    
                
                    init_img_style = min_l2_path
                    init_seg_style = min_l2_path.replace("images", "masks")
                    
                    # init_img_style = max_l2_path
                    # init_seg_style = max_l2_path.replace("images", "masks")
                
                    assert os.path.isfile(init_img_content)
                    init_image = load_img(init_img_content).to(device)
                    init_latent_content = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                    # print('latent vector shape (content img): {}'.format(init_latent_content.shape))

                    assert os.path.isfile(init_img_style)
                    init_image = load_img(init_img_style).to(device)
                    init_latent_style = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                    # print('latent vector shape (style img): {}'.format(init_latent_style.shape))

                    assert os.path.isfile(init_seg_content)
                    init_content_seg = load_img_seg(init_seg_content).to(device)

                    assert os.path.isfile(init_seg_style)
                    init_style_seg = load_img_seg(init_seg_style).to(device)
                
                    x_samples = list()
                    content_latent_size = init_latent_content.shape[2:]
                    style_latent_size   = init_latent_style.shape[2:]
                    # print(content_latent_size)
                    # print(style_latent_size)
                    latent_seg_content = F.interpolate(init_content_seg, size=content_latent_size, mode='nearest')
                    latent_seg_style   = F.interpolate(init_style_seg, size=style_latent_size, mode='nearest')

                    # new_latent = adaptive_instance_normalization_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style)
                    new_latent = mixstyle_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style, Lambda)
                    decode_imgs = vae_decode(vae_model, new_latent, scale_factor)
                    x_samples.append(decode_imgs[0])

                    for x_sample in x_samples:
                        resize_transform = transforms.Resize(target_size)
                        resized_img = resize_transform(x_sample)
                        # print(type(resized_img))
                        # print(resized_img.size())
                        # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        # img1 = Image.fromarray(x_sample.astype(np.uint8))
                        
                        # #############################################bug
                        # print('origin') # 原始图像
                        # print(img['path']) # 原始图像
                        # print('origin_mask') # 原始mask
                        # print(init_seg_content) # 原始mask
                        # print('style') # 原始图像
                        # print(init_img_style) # 风格图像
                        # print('style_mask') # 风格mask
                        # print(init_seg_style) # 风格图像
                        
                        
                        # original_path = img['path']
                        # directory, filename = os.path.split(original_path)
                        # outdir = directory.replace("images", "imagesreplace")
                        # os.makedirs(outdir, exist_ok=True)
                        # img1.save(os.path.join(outdir,str(filename)))
                        
                        # style_path = init_img_style
                        # directory_style, filename_style = os.path.split(style_path)
                        
                        # # 移动style image
                        # filename_style = filename.replace(".png", "_style.png")
                        # target_path = os.path.join(outdir,str(filename_style))
                        # # 使用 shutil.copy2() 复制文件
                        # shutil.copy2(style_path, target_path)
                        # filename_origin = filename.replace(".png", "_origin.png")
                        # target_path = os.path.join(outdir,str(filename_origin))
                        # shutil.copy2(original_path, target_path)
                        
                
                        # imgg = Image.fromarray(x_sample.astype(np.uint8))
                        # print(x_sample.size())
                        # print(osp.join(outdir, str(file)))
                        # print(type(osp.join(outdir, str(file))))
                    image_replace[i] = resized_img.unsqueeze(0)
                    
            # output = algorithm.predict(image)
            
            output_origin = algorithm.predict(image)
            output_replace = algorithm.predict(image_replace)
            output = gamma * output_origin + (1 - gamma) * output_replace
            
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def algorithm_validate_tta2_top10_test(algorithm, training_domains, data_loader, epoch, val_type):
    # nearest domain center
    
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    target_size = (224, 224)
    N = len(training_domains)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    average_values_dict = {
    'APTOS': {'mean': [0.2841766, -0.23802036, -0.44554743, 0.8143348],
              'std': [0.53095317, 0.6137561, 0.4887799, 0.3362168]},
    'MFIDDR_1view': {'mean': [0.43545985, -0.6062746, -0.01307655, 1.0771058],
                     'std': [0.73270774, 0.83608466, 0.7059133, 0.38208315]},
    'IDRID': {'mean': [0.2806504, 0.04437273, -0.75075656, 0.63489187],
              'std': [0.45809016, 0.5007796, 0.4153973, 0.32766306]},
    'RLDR': {'mean': [0.2819445, -0.46733987, -0.17754468, 0.5210681],
             'std': [0.65641, 0.820638, 0.6075364, 0.43660662]},
    'DEEPDR': {'mean': [0.40794414, -0.20940323, -0.43452525, 0.53553116],
               'std': [0.58025515, 0.6467379, 0.49657184, 0.37210035]},
    'DRTiD_1view': {'mean': [0.672681, -0.52939606, -0.31786436, 0.77008563],
                    'std': [0.52052486, 0.6000187, 0.49166602, 0.33825636]},
    'MFIDDR_4viewss': {'mean': [0.4880714, -0.5158253, 0.01185736, 1.0561396],
                       'std': [0.74223924, 0.84591156, 0.72839785, 0.37787625]},
    }
    # average_values_list = torch.Tensor([average_values_dict[domain] for domain in training_domains])
    
    new_average_values_dict = {domain: average_values_dict[domain] for domain in training_domains}
    new_average_values_list = list({'key': key, **value} for key, value in new_average_values_dict.items())

    # print(new_average_values_dict)
    all_means = []
    all_stds = []

    # get VAE (first-stage model)
    vae_path = 'v2-inference-v-first-stage-VAE.yaml'
    vae_config = OmegaConf.load(f"{vae_path}")
    vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))

    pl_sd = torch.load("checkpoints/768-v-ema-first-stage-VAE.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    vae_model.load_state_dict(sd, strict=True)

    vae_model.eval()
    vae_model = vae_model.to(device)
    scale_factor = vae_config.first_stage_config.scale_factor
    
    # 遍历每个数据集的平均值并累加
    for dataset_values in new_average_values_dict.values():
        all_means.append(dataset_values['mean'])
        all_stds.append(dataset_values['std'])
    
    type(all_means)
    # 计算所有数据集的平均值
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.mean(all_stds, axis=0)
    overall_mean_tensor = torch.tensor(overall_mean, dtype=torch.float32).view(1, -1)
    overall_std_tensor = torch.tensor(overall_std, dtype=torch.float32).view(1, -1)

    overall_mean_std = torch.cat((overall_mean_tensor, overall_std_tensor), dim=1)
    domain_mean_std = torch.cat((torch.tensor(all_means, dtype=torch.float64), torch.tensor(all_stds, dtype=torch.float64)), dim=1)

    a = 3
    # print(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = a/N*torch.sum(torch.norm(overall_mean_std - domain_mean_std, p=2, dim=1))
    threshold = 0
    
    with open('mean-std.pkl', 'rb') as file:
        mean_std = pickle.load(file) # dict list
    
    # source domain mean-std
    filtered_mean_std = [
    result_dict for result_dict in mean_std if result_dict['dataset_name'] in training_domains]
    
    # print(len(training_domains))
    
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        K = 10
        for image, label, domain, index in data_loader:
            
            image = image.cuda() # (16, 3, 224, 224)
            image_list = [image.clone() for _ in range(K)]
            label = label.cuda().long() # torch.Tensor
            
            img_mean_std = [
                next(result_dict for result_dict in mean_std if result_dict['path'] == path)
                for path in index]
            
            first_tensor_shape = torch.cat((torch.tensor(img_mean_std[0]['mean']), torch.tensor(img_mean_std[0]['std'])), dim=0).view(1, -1).shape
            content_tensor = torch.empty(0, first_tensor_shape[1])
            for result_dict in img_mean_std:
                mean__std = torch.cat((torch.tensor(result_dict['mean'], dtype=torch.float64), torch.tensor(result_dict['std'], dtype=torch.float64)), dim=0).view(1, -1)
                content_tensor = torch.cat((content_tensor, mean__std), dim=0)
            # print(content_tensor.shape) # (16, 8)
            # print(content_tensor)

            replace_indices = []
            for i, img in enumerate(img_mean_std):
                content_mean_std = torch.cat((torch.tensor(img['mean'], dtype=torch.float64), torch.tensor(img['std'], dtype=torch.float64)), dim=0).view(1, -1)
                euclid = 1/N*torch.sum(torch.norm(content_mean_std - domain_mean_std, p=2, dim=1))
                # print(euclid)
                # replace_indices.append(1 if euclid > threshold else 0)
                # replace image
                if euclid > threshold:
                    init_img_content = img['path']
                    init_seg_content = init_img_content.replace("images", "masks")
                    min_l2 = float('inf')
                    min_l2_domain = float('inf')
                    min_l2_path = None
                    min_l2_domain_name = None
                    min_l2_domain_tensor = None
                    # 计算距离最近domain
                    
                    for j, training_domain in enumerate(new_average_values_list):
                        # print(training_domain)
                        # print(type(training_domain))
                        domain_tensor = torch.cat((torch.tensor(training_domain['mean'], dtype=torch.float64), 
                                                  torch.tensor(training_domain['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        l2_domain = torch.sum(torch.norm(domain_tensor - content_mean_std, p=2, dim=1))
                        
                        # print(training_domain)
                        if l2_domain < min_l2_domain:
                            min_l2_domain = l2_domain
                            min_l2_domain_name = training_domain['key']
                            min_l2_domain_tensor = domain_tensor
                    
                    filtered2_mean_std = [
                        result_dict for result_dict in filtered_mean_std if result_dict['dataset_name']==min_l2_domain_name ]

                    topK_min_l2_paths = []
                    for j, style_info in enumerate(filtered2_mean_std):
                        style_tensor = torch.cat((torch.tensor(style_info['mean'], dtype=torch.float64), 
                                                  torch.tensor(style_info['std'], dtype=torch.float64)), dim=0).view(1, -1)
                        # l2 = torch.sum(torch.norm(style_tensor - content_mean_std, p=2, dim=1))
                        l2 = torch.sum(torch.norm(style_tensor - min_l2_domain_tensor, p=2, dim=1))
                        # 更新最小值和对应索引
                        if l2 < min_l2:
                            min_l2 = l2
                            min_l2_path = style_info['path']
                            
                        # 如果列表为空或者当前l2小于列表中最大的l2，则更新列表
                        if len(topK_min_l2_paths) < K or l2 < topK_min_l2_paths[-1][0]:
                            topK_min_l2_paths.append((l2, style_info['path']))
                            topK_min_l2_paths = sorted(topK_min_l2_paths, key=lambda x: x[0])[:10]
                        # top_min_l2_paths 现在包含了最小的10个 min_l2 和对应的 min_l2_path

                    topK_image_paths = [path for _, path in topK_min_l2_paths]
                    # print(min_l2_path)
                    # print(top10_image_paths)
                    
                    for l, image_paths in enumerate(topK_image_paths):
                        init_img_style = image_paths
                        init_seg_style = image_paths.replace("images", "masks")
                
                        assert os.path.isfile(init_img_content)
                        init_image = load_img(init_img_content).to(device)
                        init_latent_content = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                        # print('latent vector shape (content img): {}'.format(init_latent_content.shape))

                        assert os.path.isfile(init_img_style)
                        init_image = load_img(init_img_style).to(device)
                        init_latent_style = get_vae_encoding(vae_model.encode(init_image), scale_factor)  # move to latent space
                        # print('latent vector shape (style img): {}'.format(init_latent_style.shape))

                        assert os.path.isfile(init_seg_content)
                        init_content_seg = load_img_seg(init_seg_content).to(device)

                        assert os.path.isfile(init_seg_style)
                        init_style_seg = load_img_seg(init_seg_style).to(device)
                
                        x_samples = list()
                        content_latent_size = init_latent_content.shape[2:]
                        style_latent_size   = init_latent_style.shape[2:]
                        # print(content_latent_size)
                        # print(style_latent_size)
                        latent_seg_content = F.interpolate(init_content_seg, size=content_latent_size, mode='nearest')
                        latent_seg_style   = F.interpolate(init_style_seg, size=style_latent_size, mode='nearest')

                        # new_latent = adaptive_instance_normalization_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style)
                        new_latent = mixstyle_with_mask(init_latent_content, latent_seg_content, init_latent_style, latent_seg_style)
                        decode_imgs = vae_decode(vae_model, new_latent, scale_factor)
                        x_samples.append(decode_imgs[0])

                        for x_sample in x_samples:
                            resize_transform = transforms.Resize(target_size)
                            resized_img = resize_transform(x_sample)
                        
                        image_list[l][i] = resized_img.unsqueeze(0)
                        # image[i] = resized_img.unsqueeze(0)
                    
            all_outputs = []
            for image_K in image_list:
                output = algorithm.predict(image_K)
                all_outputs.append(output)
            
            all_outputs_tensor = torch.stack(all_outputs)
            average_output = torch.mean(all_outputs_tensor, dim=0)
            # print(average_output.size())
            # output = algorithm.predict(image)
            output = average_output
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)


    algorithm.train()
    return auc_ovo, loss

def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    # evaluate on each severity and type of corruption in turn
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")