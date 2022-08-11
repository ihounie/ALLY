import numpy as np
import sys
import gzip
import gc
import os
import argparse
from dataset import get_dataset, get_handler
import models
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from strategy import Strategy
import random
from ally import ALLYSampling
from baselines import BadgeSampling, RandomSampling
import wandb

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# data defaults
args_pool = {
            'MNIST':
                {
                 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 3},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 3}
                },
            'SVHN':
                {
                 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 3},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 3}
                },
            'CIFAR10':
                {
                 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 3},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 3}
                },
            'STL10':
                {
                 'transform': transforms.Compose([transforms.ToTensor()]), #  if unnormalized add: transforms.Normalize(mean=[114.06, 112.23, 103.62], std=[66.40, 65.411, 68.98])]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 3},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 3}
                }
            }

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='resnet')
parser.add_argument('--path', help='data path', type=str, default='./data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=200)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=10000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
parser.add_argument('--epsilon', help='constant tightness', type=float, default=0.2)
parser.add_argument('--nPrimal', help='number of primal steps', type=int, default=1)
parser.add_argument('--nPat', help = 'es epochs before halt cond', type = int, default = 1)
parser.add_argument('--lr_dual', help='number of dual steps', type=float, default=0.05)
parser.add_argument('--seed', help='seed', type=int, default=1357)
parser.add_argument('--cluster', help='How to cluster for diversity in primaldual', type = str, default='nocluster')
parser.add_argument('--projname', help='Project name for wandb', type = str, default='AProjectHasNoName')
parser.add_argument('--lambdaTestSize', help = 'Size in percentage of test set for lambda net', type = float, default = 0)
parser.add_argument('--name', help='name for wandb', type=str, default="a run has no name")
opts = parser.parse_args()

seed_everything(opts.seed)


# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB)/ opts.nQuery)
DATA_NAME = opts.data            

opts.nClasses = 10
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']
args_pool['CIFAR10']['transformTest'] = args_pool['CIFAR10']['transform']
args_pool['STL10']['transformTest'] = args_pool['STL10']['transform']
args = args_pool[DATA_NAME]
args['lr'] = opts.lr
args['nPrimal'] = opts.nPrimal
args['seed'] = opts.seed
args['nClasses'] = opts.nClasses
args['nPat'] = opts.nPat
args['alg'] = opts.alg


if not os.path.exists(opts.path):
    os.makedirs(opts.path)

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path)
opts.dim = np.shape(X_tr)[1:]
handler = get_handler(opts.data)

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

if opts.model == 'mlp':
    net = models.mlp(opts.dim, embSize=opts.nEmb)
elif opts.model == 'resnet':
    if opts.data != "STL10":
        net = models.ResNet18()
    else:
        net = models.ResNet9()
else: 
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

if args["alg"] == "ALLY":
    alg = ALLYSampling(X_tr, Y_tr, idxs_lb, net, handler, args, opts.epsilon, opts.cluster, opts.lr_dual, opts.nPrimal, opts.lambdaTestSize)                                                            
elif args["alg"] == "random":
    alg = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif args["alg"] == "badge":
    alg = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)

print(DATA_NAME, flush=True)
print(type(alg).__name__, flush=True)

# Initialize active learning strategy
alg.train()
P = alg.predict(X_te, Y_te)
probs = alg.predict_prob(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
loss = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
loss[0] = F.cross_entropy(probs, Y_te).item()
print(f"\n\nNumber of samples = {sum(idxs_lb)} ------> Testing accuracy: {acc[0]} , Loss: {loss[0]} \n\n", flush=True)

sampled = []

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd), flush=True)
    torch.cuda.empty_cache()
    gc.collect()

    # Query
    output = alg.query(NUM_QUERY)
    q_idxs = output
    sampled += list(q_idxs)
    idxs_lb[q_idxs] = True

    # Update
    alg.update(idxs_lb)
    alg.train()

    # Evaluate round accuracy
    P = alg.predict(X_te, Y_te)
    probs = alg.predict_prob(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    loss[rd] = F.cross_entropy(probs, Y_te).item()
    print(f"\n\nNumber of samples = {sum(idxs_lb)} ------> testing accuracy: {acc[rd]} , loss: {loss[rd]} \n\n", flush=True)
    nsamples = NUM_INIT_LB + rd*NUM_QUERY

    if sum(~alg.idxs_lb) < opts.nQuery: 
        sys.exit('Too few remaining points to query')

print(f"\nAccuracy evolution: {acc}")
print(f"\nCross Entropy evolution: {loss}")

