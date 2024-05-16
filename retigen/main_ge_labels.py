import algorithms
import os
from utilsGDRBench.validate import *
from utilsGDRBench.args import *
from utilsGDRBench.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm


if __name__ == "__main__":

    
    args = get_args()
    cfg = setup_cfg(args)
    log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)

    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.num_classes, cfg)
    algorithm.cuda()
    algorithm.renew_model(log_path)
    
    print(cfg.ALGORITHM)
    print(cfg.DATASET.TARGET_DOMAINS)

    _, test_auc = algorithm.generate_pseudo_labels(val_loader, test_loader)
    print('test_auc:{}'.format(test_auc))
    
    

