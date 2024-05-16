from . import fundusaug as FundusAug
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

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


# FundusAug contains ["RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter", "Sharpness", "Halo", "Hole", "Spot", "Blur"],
# these operations are splited into pre_FundusAug and post_FundusAug

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