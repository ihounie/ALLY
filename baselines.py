import numpy as np
from torch.utils.data import DataLoader
from strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# kmeans ++ initialization for BADGE
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "badge"

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]

class RandomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(RandomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "random"

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]

class EntropySampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(EntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.alg = "entropy"

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]

class CoreSetSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, tor=1e-4):
        super(CoreSetSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.tor = tor
        self.alg = "coreset"

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen]


    def query_old(self, n):
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        print('calculate distance matrix')
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        print('calculate greedy solution')
        mat = dist_mat[~lb_flag, :][:, lb_flag]

        for i in range(n):
            if i % 10 == 0:
                print('greedy solution {}/{}'.format(i, n))
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
            lb_flag[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        opt = mat.min(axis=1).max()

        bound_u = opt
        bound_l = opt/2.0
        delta = opt

        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        lb_flag_ = self.idxs_lb.copy()
        subset = np.where(lb_flag_==True)[0].tolist()

        sols = None

        if sols is None:
            q_idxs = lb_flag
        else:
            lb_flag_[sols] = True
            q_idxs = lb_flag_
        print('sum q_idxs = {}'.format(q_idxs.sum()))

        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]

def select(X, K, fisher, iterates, lamb=1, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    # forward selection, over-sample by 2x
    print('forward selection...', flush=True)
    over_sample = 2
    for i in range(int(over_sample *  K)):

        # check trace with low-rank updates (woodbury identity)
        xt_ = X.cuda() 
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

        # clear out gpu memory
        xt = xt_.cpu()
        del xt, innerInv
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # get the smallest unselected item
        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break

        indsAll.append(ind)
        print(i, ind, traceEst[ind], flush=True)
       
        # commit to a low-rank update
        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

    # backward pruning
    print('backward pruning...', flush=True)
    for i in range(len(indsAll) - K):

        # select index for removal
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()
        print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


        # low-rank update (woodbury identity)
        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll

class BaitSampling(Strategy):
    
    def __init__(self, X, Y, idxs_lb, net, handler, args, lamb = 1e-2):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)        
        self.lamb = lamb
        self.alg = "BAIT"

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # get low-rank point-wise fishers
        xt = self.get_exp_grad_embedding(self.X, self.Y)

        # get fisher
        print('getting fisher matrix...', flush=True)
        batchSize = 1000 # should be as large as gpu memory allows
        nClass = torch.max(self.Y).item() + 1
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(len(self.X) / batchSize))
        for i in range(int(np.ceil(len(self.X) / batchSize))):
            xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        nClass = torch.max(self.Y).item() + 1
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        rounds = int(np.ceil(len(xt2) / batchSize))
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        chosen = select(xt[idxs_unlabeled], n, fisher, init, lamb=self.lamb, nLabeled=np.sum(self.idxs_lb))
        return idxs_unlabeled[chosen]

    # fisher embedding for bait (assumes cross-entropy loss)
    def get_exp_grad_embedding(self, X, Y, probs=[], model=[]):
        if type(model) == list:
            model = self.clf

        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))

        embedding = np.zeros([len(Y), nLab, embDim * nLab])
        for ind in range(nLab):
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    cout, out = model(x)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                        if len(probs) > 0: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[idxs[j]][ind])
                        else: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
        return torch.Tensor(embedding)
