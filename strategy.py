import numpy as np
from torch import nn
import random
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
from torch.utils.data.dataset import TensorDataset

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.nPat = args["nPat"]
        use_cuda = torch.cuda.is_available()

        #Create train and validation idxs
        num_val = int(sum(idxs_lb)*0.15)
        self.idxs_val = np.arange(self.n_pool)[~self.idxs_lb]
        np.random.shuffle(self.idxs_val)
        self.idxs_val = self.idxs_val[:num_val]

        self.idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        self.idxs_train = [i for i in self.idxs_train if i not in self.idxs_val]

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        # Update train and validation idxs
        num_val = int(sum(idxs_lb)*0.15)
        self.idxs_val = np.arange(self.n_pool)[~self.idxs_lb]
        np.random.shuffle(self.idxs_val)
        self.idxs_val = self.idxs_val[:num_val]

        self.idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        self.idxs_train = [i for i in self.idxs_train if i not in self.idxs_val]

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()

        return loss.item(), accFinal / len(loader_tr.dataset.X)

    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.clf =  self.net.apply(weight_reset).cuda()
        print(f"Learning Rate {self.args['lr']}")
        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        loader_tr = DataLoader(self.handler(self.X[self.idxs_train], torch.Tensor(self.Y.numpy()[self.idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        # Reset lambdas at beginning of each round
        self.lambdas = np.zeros(len(self.idxs_train))

        epoch = 1
        accCurrent = 0.
        accBest = 0.
        best_model = None

        epochs_no_improve = 0
        early_stop = False
  
        while accCurrent < 0.99 and not early_stop:
            
            if self.alg == "ALLY":
                lossCurrent, accCurrent = self._PDCL(epoch, loader_tr, optimizer)
            else:
                lossCurrent, accCurrent = self._train(epoch, loader_tr, optimizer)

            if (epoch % 50 == 0) and (accCurrent < 0.2): # reset if not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
            
            val_acc = self.validate()

            if val_acc >= accBest:
                accBest = val_acc
                epochs_no_improve = 0
                best_model = deepcopy(self.clf)
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve > self.nPat:
                early_stop = True

            print(f"{epoch} training accuracy: {accCurrent:.2f} \tTraining loss: {lossCurrent:.2f} \tValidation acc: {val_acc:.2f}", flush=True)
            epoch += 1   

        self.clf = best_model
            
    def validate(self):
        X, Y = self.X[self.idxs_val], torch.Tensor(self.Y.numpy()[self.idxs_val]).long()
        preds = self.predict(X, Y)
        acc = 1.0 * (Y == preds).sum().item() / len(Y)
        return acc

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()        
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y):
        
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[]):
        if type(model) == list:
            model = self.clf
        
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)