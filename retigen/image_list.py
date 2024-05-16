import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img


class RetiGenBench(Dataset):
    
    def __init__(self, root, source_domains, 
                 target_domains, mode, trans_basic=None, 
                 trans_mask = None, trans_fundus=None, 
                 transform=None, pseudo_item_list=None):
        root = osp.abspath(osp.expanduser(root))
        print('#######################')
        print(root)
        # root = '/media/raid/gongyu/projects/MVDRG/GDRBench_Data'
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.dataset_tta_dir = osp.join(root, "imagestta")
        self.split_dir = osp.join(root, "splits")
        

        self.data = []
        self.label = []
        self.domain = []
        self.masks = []
        
        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask
        self.transform = transform

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            # self._read_data(target_domains, "test")
            A = self._read_data(target_domains, "test")
            print("GDRBench--item_list--start")
            self.item_list = (
                pseudo_item_list if pseudo_item_list else A)
            print('-------read_data-----finished')
            print(len(self.item_list))
        elif mode == "balance":
            # self._read_data(target_domains, "test")
            A = self._read_data(target_domains, "balance")
            print("GDRBench--item_list--start")
            self.item_list = (
                pseudo_item_list if pseudo_item_list else A)
            print('-------read_data-----finished')
            print(len(self.item_list))
        elif mode == "pseudo_blc":
            # self._read_data(target_domains, "test")
            A = self._read_data(target_domains, "pseudo_blc")
            print("GDRBench--item_list--start")
            self.item_list = A
            print('-------read_data-----finished')
            print(len(self.item_list))
        elif mode == "tta":
            self._read_data(target_domains, "tta")
        
        
    def _read_data(self, input_domains, split):
        item_list = []
        for domain, dname in enumerate(input_domains):
            if split == "test":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split(file_val)
                # file_path = "/media/raid/gongyu/projects/MVDRG/GDRBench_Data/splits/Ablation_MFIDDR_4viewss_source_APTOS_ERM_pseudo.txt"
                #file_path = "/media/raid/gongyu/projects/MVDRG/GDRBench_Data/splits/Ablation_MFIDDR_4viewss_source_APTOS_GDRNet_truth_filtered.txt"
                #impath_label_list = self._read_split(file_path)
            elif split == "balance":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split_balance(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split_balance(file_val)
            elif split == "pseudo_blc":
                file_train = osp.join(self.split_dir, dname + "_pseudo.txt")
                impath_label_list = self._read_split_pseudo_blc(file_train)
            elif split == "tta":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split_tta(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split_tta(file_val)
            else:
                file = osp.join(self.split_dir, dname + "_" + split + ".txt")
                impath_label_list = self._read_split(file)

            for impath, label in impath_label_list:
                self.data.append(impath)
                self.masks.append(impath.replace("images", "masks"))

                self.label.append(label)
                self.domain.append(domain)
                item_list.append((impath, label, impath))
        return item_list

    def _read_split(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))
                
        return items
    
    def _read_split_balance(self, split_file):
        class_count = {}  # 用于统计每个类别的样本数量
        items = []

        # 读取split_file文件，统计每个类别的样本数量并存储为字典
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))
                class_count[label] = class_count.get(label, 0) + 1

        # 获取最小类别样本数
        min_samples = min(class_count.values())

        # 构建平衡数据集，抽取每一类label最前面的min_samples张图片
        items_balance = []
        class_samples = {label: 0 for label in class_count}
        for impath, label in items:
            if class_samples[label] < min_samples:
                items_balance.append((impath, label))
                class_samples[label] += 1

        return items_balance
    def _read_split_pseudo_blc(self, split_file):
        class_count = {}  # 用于统计每个类别的样本数量
        items = []

        # 读取split_file文件，统计每个类别的样本数量并存储为字典
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))
                class_count[label] = class_count.get(label, 0) + 1

        # 获取最小类别样本数
        min_samples = min(class_count.values())

        # 构建平衡数据集，抽取每一类label最前面的min_samples张图片
        items_balance = []
        class_samples = {label: 0 for label in class_count}
        random.seed(2024)
        random.shuffle(items)
        for impath, label in items:
            if class_samples[label] < min_samples:
                items_balance.append((impath, label))
                class_samples[label] += 1

        return items_balance    
    
    def _read_split_tta(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_tta_dir, impath)
                label = int(label)
                items.append((impath, label))
                
        return items

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # print(index)
        path = self.data[index]
        
        data = Image.open(self.data[index]).convert("RGB")

        if self.mode == "train":
            mask = Image.open(self.masks[index]).convert("L")

        label = self.label[index]
        domain = self.domain[index]
        
        # if self.trans_basic is not None:
        #     data = self.trans_basic(data)
        
        if self.trans_mask is not None:
            mask = self.trans_mask(mask)
            
        if self.transform is not None:
            # combined_transforms = transforms.Compose([
            #     self.trans_basic,
            #     self.transform
            #     ])
            data = self.transform(data)
        else:
            data = self.trans_basic(data)

        ####################################
        # index = path
        
        if self.mode == "train":
            return data, mask, label, domain, index
        else:
            # return data, label, domain, index
            return data, label, index, path
        
class RetiGenBench_pseudo_blc(Dataset):
    
    def __init__(self, root, source_domains, 
                 target_domains, mode, trans_basic=None, 
                 trans_mask = None, trans_fundus=None, 
                 transform=None, Args=None):
        root = osp.abspath(osp.expanduser(root))
        print(root)
        # root = '/media/raid/gongyu/projects/MVDRG/GDRBench_Data'
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")
        
        self.data = []
        self.label = []
        self.domain = []
        self.masks = []
        
        self.trans_basic = trans_basic
        self.trans_fundus = trans_fundus
        self.trans_mask = trans_mask
        self.transform = transform

        algorithm_name = Args.model_src.algorithm
        dg_mode = Args.dg_mode
        MVLCE = Args.mvlce
        if MVLCE:
            if dg_mode == 'DG': 
                file_path = osp.join(self.split_dir, "MVLCE_" + str(dg_mode) + "_" + str(target_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
                print(file_path)
            elif dg_mode == 'ESDG':
                file_path = osp.join(self.split_dir, "MVLCE_" + str(dg_mode) + "_" + str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
                print(file_path)
            else:
                print("wrong dg_mode")
        else:
            if dg_mode == 'DG': 
                file_path = osp.join(self.split_dir, str(dg_mode) + "_" + str(target_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
                print(file_path)
            elif dg_mode == 'ESDG':
                file_path = osp.join(self.split_dir, str(dg_mode) + "_" + str(target_domains[0]) + "_source_" + str(source_domains[0]) + "_" +  algorithm_name + "_pseudo.txt")
                print(file_path)
            else:
                print("wrong dg_mode")
        # file_path = "/media/raid/gongyu/projects/MVDRG/GDRBench_Data/splits/Ablation_MFIDDR_4viewss_source_APTOS_GDRNet_pseudo_filtered.txt"
        print(file_path)
        if mode == "pseudo_blc":
            # self._read_data(target_domains, "test")
            A = self._read_data(target_domains, file_path, "pseudo_blc", Args)
            print("GDRBench--item_list--start")
            self.item_list = A
            print('-------read_data-----finished')
            print(len(self.item_list))
        else:
            print("wrong!")
        
        
    def _read_data(self, input_domains, file_train, split, args):
        item_list = []
        for domain, dname in enumerate(input_domains):
            if split == "pseudo_blc":
                # file_train = osp.join(self.split_dir, dname + "_pseudo.txt")
                impath_label_list = self._read_split_pseudo_blc(file_train, args)
            else:
                print("wrong!")

            for impath, label in impath_label_list:
                self.data.append(impath)
                self.masks.append(impath.replace("images", "masks"))

                self.label.append(label)
                self.domain.append(domain)
                item_list.append((impath, label, impath))
        return item_list
    
    def _read_split_pseudo_blc(self, split_file, args):
        class_count = {}  # 用于统计每个类别的样本数量
        class_count_patient_level = {}
        items = []
        patients = []
        patients_balance = []
        patients_id_balance = []
        views = 4
        # 读取split_file文件，统计每个类别的样本数量并存储为字典
        with open(split_file, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                file_info = line.split("images/")[1]
                line = osp.join(self.dataset_dir, file_info)
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))
                if (idx + 1) % views == 1:  # 第1、5、9行...
                    patients.append((impath, label))
                    class_count_patient_level[label] = class_count_patient_level.get(label, 0) + 1
                class_count[label] = class_count.get(label, 0) + 1
        
        seed = args.seed
        random.seed(seed)
        
        MVLCE = args.mvlce
        if MVLCE:
            # patient level down sampling
            random.shuffle(patients)
            min_patients = min(class_count_patient_level.values())
            #min_patients = 20
        
            class_samples = {label: 0 for label in class_count}
            for impath, label in patients:
                if class_samples[label] < min_patients:
                    patients_balance.append((impath, label))
                    class_samples[label] += 1
        
            items_balance = []
            for imp, lab in patients_balance:
                patient_id = imp.split('/')[-1].split('_')[:-1]
                patient_id = '_'.join(patient_id)
                for impath, label in items:
                    if patient_id in impath:
                        items_balance.append((impath, label))
            print(len(items_balance))
        else:
            # image level down sampling
            print(patients_id_balance)
            # 获取最小类别样本数
            min_samples = min(class_count.values())
            # min_samples = 100
            # 构建平衡数据集，抽取每一类label最前面的min_samples张图片
            items_balance = []
            class_samples = {label: 0 for label in class_count}
            random.seed(seed)
            random.shuffle(items)
            for impath, label in items:
                if class_samples[label] < min_samples:
                    items_balance.append((impath, label))
                    class_samples[label] += 1

        return items_balance    
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaa')
        # print(index)
        path = self.data[index]
        
        data = Image.open(self.data[index]).convert("RGB")

        if self.mode == "train":
            mask = Image.open(self.masks[index]).convert("L")

        label = self.label[index]
        domain = self.domain[index]
        
        # if self.trans_basic is not None:
        #     data = self.trans_basic(data)
        
        if self.trans_mask is not None:
            mask = self.trans_mask(mask)
            
        if self.transform is not None:
            # combined_transforms = transforms.Compose([
            #     self.trans_basic,
            #     self.transform
            #     ])
            data = self.transform(data)
        else:
            data = self.trans_basic(data)

        ####################################
        # index = path
        
        if self.mode == "train":
            return data, mask, label, domain, index
        else:
            # return data, label, domain, index
            return data, label, index, path