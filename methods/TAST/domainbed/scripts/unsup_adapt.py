# The code is modified from domainbed.scripts.train

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy

from torchvision import transforms
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def generate_featurelized_loader(loader, network, classifier, batch_size=32):
    """
    The classifier adaptation does not need to repeat the heavy forward path, 
    We speeded up the experiments by converting the observations into representations. 
    """
    z_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)
        
        z_list.append(z.detach().cpu())
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())
        # p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader1, loader2, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def accuracy_ent(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0
    label_list = []
    output_list = []
    pred_list = []
    softmax = torch.nn.Softmax(dim=1)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # print(x.shape) # torch.Size([32, 3, 224, 224])
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            _, pred = torch.max(p, 1)
            output_sf = softmax(p)
            
            label_list.append(y.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
            
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        # print(f"Accuracy: {acc}")
        # print(f"F1 Score (Macro): {f1}")
        # print(f"AUC (OVO, Macro): {auc_ovo}")
        
    network.train()

    return correct / total, ent / total, auc_ovo, acc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--adapt_algorithm', type=str, default="T3A")
    args_in = parser.parse_args()

    epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
    records = []
    
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    print(args)
    args.input_dir = args_in.input_dir

    if '-' in args_in.adapt_algorithm:
        args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split('-')
        args.test_batch_size = int(test_batch_size)
    else:
        args.adapt_algorithm = args_in.adapt_algorithm
        args.test_batch_size = 32  # default

    args.output_dir = args.input_dir
    
    alg_name = args_in.adapt_algorithm

    if args.adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf', 'TAST']:
        use_featurer_cache = True
    else:
        use_featurer_cache = False
    if os.path.exists(os.path.join(args.output_dir, 'done_{}'.format(alg_name))):
        print("{} has already excecuted".format(alg_name))

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    algorithm_dict = None
    # os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out_{}.txt'.format(alg_name)))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err_{}.txt'.format(alg_name)))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    assert os.path.exists(os.path.join(args.output_dir, 'done'))
    # assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    if args.dataset in vars(datasets):
        # print(vars(datasets)[args.dataset])
        # dataset = vars(datasets)[args.dataset](args.data_dir,
        #     args.test_envs, hparams, test_ts)
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams, args)
        print(args.test_envs) # 4 [MFIDDR]
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    all_splits = []
    print(len(dataset))
    for env_i, env in enumerate(dataset):
        # print('!!!!!!!!!!!!!!!!!!!!!!!')
        if env_i not in args.test_envs:
            continue
        print(env)
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        _, all = misc.split_dataset(env,
            int(len(env)*0),
            misc.seed_hash(args.trial_seed, env_i))
        
        if env_i in args.test_envs:
            # print('!!!!!!!!!!!!!!!!!!!!!!!')
            # print(env)
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            # print(len(uda))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        # print(in_weights)
        # print(out_weights)
        # print(uda_weights)
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        all_splits.append((all, None))
        if len(uda):
            uda_splits.append((uda, uda_weights))
        print(len(in_)) # 6890
        print(len(out)) # 1722
        print(len(all)) # 1722

    # Use out splits as training data (to fair comparison with train.py)
    train_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.test_batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    
    test_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.test_batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (all_splits)]
    
    print(len(eval_loaders))
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    test_weights = [None for _, weights in (all_splits)]

    # eval_loader_names = ['env{}_in'.format(i)
    #     for i in range(len(in_splits))]
    # eval_loader_names += ['env{}_out'.format(i)
    #     for i in range(len(out_splits))]
    # eval_loader_names += ['env{}_uda'.format(i)
    #     for i in range(len(uda_splits))]
    eval_loader_names = ['env{}_in'.format(4)]
    eval_loader_names += ['env{}_out'.format(4)]
    eval_loader_names += ['env{}_uda'.format(4)]
    test_loader_names = ['env{}_all'.format(4)]
    # print(len(in_splits))
    # print(len(out_splits))
    # print(len(uda_splits))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # load trained model
    log_path = args.output_dir
    print(args.algorithm)
    if args.algorithm == 'ERM' or args.algorithm == 'GDRNet':
        algorithm.renew_model(log_path)
    # ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    # algorithm_dict = ckpt['model_dict']
    # if algorithm_dict is not None:
    #     algorithm.load_state_dict(algorithm_dict)

    # Evaluate base model
    print("Base model's results")
    results = {}
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    tests = zip(test_loader_names, test_loaders, test_weights)
    
    # for name, loader, weights in evals:
    #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    #     acc, ent = accuracy_ent(algorithm, loader, weights, device, adapt=None)
    #     results[name+'_acc'] = acc
    #     results[name+'_ent'] = ent
    # results_keys = sorted(results.keys())
    
    for name, loader, weights in tests:
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        acc, ent, Auc, Acc, f1 = accuracy_ent(algorithm, loader, weights, device, adapt=None)
        results[name+'_acc_all'] = acc
        results[name+'_ent_all'] = ent
        print(f"Accuracy: {Acc}")
        print(f"F1 Score (Macro): {f1}")
        print(f"AUC (OVO, Macro): {Auc}") 
    results_keys = sorted(results.keys())
    
    # misc.print_row(results_keys, colwidth=12)
    # misc.print_row([results[key] for key in results_keys], colwidth=12)

    print("\nAfter {}".format(alg_name))
    # Cache the inference results
    if use_featurer_cache:
        original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
        loaders = []
        for name, loader, weights in original_evals:
            loader1, loader2, ent = generate_featurelized_loader(loader, network=algorithm.featurizer, classifier=algorithm.classifier, batch_size=32)
            loaders.append((name, loader1, weights))
    else:
        loaders = zip(eval_loader_names, eval_loaders, eval_weights)
    
    evals = []
    for name, loader, weights in loaders:
        print(name, loader, weights)
        if name in ['env{}_in'.format(i) for i in args.test_envs]:
            train_loader = (name, loader, weights)
        else:
            evals.append((name, loader, weights))
    print(evals)
    last_results_keys = None
    adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
        args.adapt_algorithm)
    
    if args.adapt_algorithm in ['T3A']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20, 50, 100, -1], 
        }
    elif args.adapt_algorithm in ['TentFull', 'TentPreBN', 'TentClf', 'TentNorm']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3]
        }
    elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9]
        }
    elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9], 
            'theta': [0.1], 
        }
    elif args.adapt_algorithm in ['TAST_BN']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
        }
    elif args.adapt_algorithm in ['TAST']:
        adapt_hparams_dict = {
            'num_ensemble': [1, 5, 10, 20],
            'filter_K': [1, 5, 20, 50, 100, -1],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
            'init_mode': ['kaiming_normal']
        }
    else:
        raise Exception("Not Implemented Error")
    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]

    for adapt_hparams in adapt_hparams_list:
        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), adapt_hparams, algorithm
        )
        # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
        adapted_algorithm.to(device)
        
        results = adapt_hparams

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        # ## Usual evaluation
        for name, loader, weights in evals:
            acc, ent, auc, Acc, f1 = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            adapted_algorithm.reset()

        name, loader, weights = train_loader
        acc, ent, Auc, Acc, f1 = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent
        results[name+'_Auc'] = Auc
        results[name+'_Acc'] = Acc
        results[name+'_f1'] = f1
        print(f"Accuracy: {Acc}")
        print(f"F1 Score (Macro): {f1}")
        print(f"AUC (OVO, Macro): {Auc}")
        

        # tests = zip(test_loader_names, test_loaders, test_weights)
        # for name, loader, weights in tests:
        #     print('???????????????????????????')
        #     acc, ent, Auc, Acc, f1 = accuracy_ent(adapted_algorithm, loader, weights, device, adapt=True)
        #     results[name+'_acc_all'] = acc
        #     results[name+'_ent_all'] = ent
        #     results[name+'_Auc_all'] = Auc
        #     results[name+'_Acc_all'] = Acc
        #     results[name+'_F1_all'] = f1
        #     print(f"Accuracy: {Acc}")
        #     print(f"F1 Score (Macro): {f1}")
        #     print(f"AUC (OVO, Macro): {Auc}") 
        
        del adapt_hparams['cached_loader']
        results_keys = sorted(results.keys())

        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys],
            colwidth=12)
        
            
        results.update({
            'hparams': hparams,
            'args': vars(args)    
        })
        # save file
        epochs_path = os.path.join(args.output_dir, 'results_{}.jsonl'.format(alg_name))
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

    # create done file
    with open(os.path.join(args.output_dir, 'done_{}'.format(alg_name)), 'w') as f:
        f.write('done')

        
