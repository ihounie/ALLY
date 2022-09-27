import numpy as np
from sklearn.model_selection import train_test_split
import torch
from strategy import Strategy
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from torch.utils.data.dataset import TensorDataset
import pdb
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy import stats
from lambdautils import lambdanet, lambdaset

class ALLYSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, epsilon = 0.2, cluster = 'kmeans', lr_dual = 0.05, nPrimal = 1, lambda_test_size = 0, nPat = 2, dlr = 0.98):
        super(ALLYSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        
        #self.lambdas = np.zeros(sum(self.idxs_lb))
        self.lambdas = np.ones(sum(self.idxs_lb))
 
        self.seed = args["seed"]
        self.nClasses = args["nClasses"]
        self.nPat = nPat
        self.epsilon = epsilon
        self.lr_dual = lr_dual
        self.dlr = dlr
        self.cluster = cluster
        self.nPrimal = nPrimal # Not used in minimal version with alternate primaldual (nPrimal = 1)
        self.lambda_test_size = lambda_test_size
        self.alg = "ALLY"
                
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_lb = np.arange(self.n_pool)[self.idxs_lb]

        # Prepare data
        print("Generating Embdeddings...")
        X_train, X_test, y_train, y_test = self.prepare_data_lambda(self.X[idxs_lb], self.Y.numpy()[idxs_lb])

        # Train Lambdanet
        self.reg = lambdanet(input_dim = self.net.get_embedding_dim()).cuda()
        self.train_test_lambdanet(X_train, X_test, y_train, y_test)

        # Predict on unlabeled samples
        X_embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        preds = self.predict_lambdas(X_embedding)
        self.lambdas_pred = preds
        
        # Sort samples by lambda
        idxs_lambdas_descending = (-preds).argsort()
        
        # Select samples with highest predicted lambda from each cluster
        if self.cluster == "kmeans":
            # MiniBatch K-means on embeddings
            print("Clustering....")
            nClusters = n
            kmeans = MiniBatchKMeans(n_clusters = nClusters, random_state = self.seed, batch_size=1024)
            cluster_idxs = kmeans.fit_predict(X_embedding)
    
            # Select highest lambdas from each cluster
            chosen = []
            space_in_clust = np.zeros(nClusters)+n//nClusters
            for sample_idx in idxs_lambdas_descending:
                if space_in_clust[cluster_idxs[sample_idx]] > 0:
                    chosen.append(sample_idx)
                    space_in_clust[cluster_idxs[sample_idx]] -= 1
                if len(chosen) >= n:
                    break     
            
        # No diversity
        else:
            chosen = idxs_lambdas_descending[:n]
        
        print("Done selecting new batch.")
        return idxs_unlabeled[chosen]

    def prepare_data_lambda(self, X, Y):
        X_embedding = self.get_embedding(X, Y).numpy()
        y_lambdas = self.lambdas
        if self.lambda_test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X_embedding, y_lambdas, test_size=self.lambda_test_size, random_state = self.seed)
        else:
            X_train = X_embedding
            X_test = []
            y_train = y_lambdas
            y_test = []
        return X_train, X_test, y_train, y_test

    def _train_lambdanet(self, epoch, loader_tr, optimizer, scheduler):
        self.reg.train()
        mseFinal = 0.

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda().float()), Variable(y.cuda().float())
            optimizer.zero_grad()
            out = self.reg(x)
            loss = F.mse_loss(out.squeeze(), y)
            loss.backward()
            mseFinal += loss.item()
            optimizer.step()
        scheduler.step()
        
        return mseFinal/len(loader_tr)

    def train_test_lambdanet(self, X_train, X_test, y_train, y_test):

        optimizer = optim.Adam(self.reg.parameters(), lr = 0.0025, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.95)

        loader_tr = DataLoader(lambdaset(X_train, X_test, y_train, y_test, train = True), batch_size = 64, shuffle = False, drop_last=True)

        mseThresh = 1e-3 #Add as argument
        #Train
        self.reg.train()
        epoch = 1
        mseCurrent = 10.
        print_every = 10
        while (mseCurrent > mseThresh) and (epoch < 150): #default values for STL
            mseCurrent = self._train_lambdanet(epoch, loader_tr, optimizer, scheduler)
            if epoch%print_every==0:
                print(f"{epoch} Lambda training mse:  {mseCurrent:.3f}", flush=True)
            epoch += 1
               
        mseFinal = 0.

        # Test L if needed
        if self.lambda_test_size > 0:
            P = self.predict_lambdas(X_test, y_test)
            mseTest = F.mse_loss(P, torch.tensor(y_test))           
            print(f"-----> Lambda test mse: {mseTest.item():.2f}\n", flush=True)
        return None
	

    def predict_lambdas(self, X, Y=None):
        
        if Y is None:
            Y = np.zeros(len(X))
        loader_te = DataLoader(lambdaset(None, X, None, Y, train = False), batch_size = 64, shuffle = False, drop_last=True)

        self.reg.eval()       
        P = torch.zeros(len(Y))
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda().float()), Variable(y.cuda().float())
                out = self.reg(x)
                P[idxs] = out.squeeze().data.cpu()
        return P

    def _PDCL(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        lossCurrent = 0.

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            
            # Snapshot of current dual variables
            lambdas = self.lambdas[idxs]
            lambdas = torch.tensor(lambdas, requires_grad = False).cuda()

            # Primal Update (assuming nPrimal=1 and \ell = \ell')
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x)

            # Compute Lagrangian
            loss = F.cross_entropy(out, y, reduction = 'none')
            lossCurrent += torch.mean(loss).item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            lagrangian = torch.mean(loss*(1+lambdas)-lambdas*self.epsilon)
            
            # Step to minimize Lagrangian
            lagrangian.backward()
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.15, max=.15)
            
            # Update params
            optimizer.step()

            # Compute Slack and perform Dual Update 
            lambdas += self.lr_dual*(loss-self.epsilon) 
            lambdas[lambdas < 0] = 0
            self.lambdas[idxs] = lambdas.detach().cpu()

        return lossCurrent/len(loader_tr), accFinal/len(loader_tr.dataset.X)

    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.clf =  self.net.apply(weight_reset).cuda()
        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        loader_tr = DataLoader(self.handler(self.X[self.idxs_train], torch.Tensor(self.Y.numpy()[self.idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        # Reset lambdas at beginning of each round
        #self.lambdas = np.zeros(len(self.idxs_train))
        self.lambdas = np.ones(len(self.idxs_train))


        epoch = 1
        accCurrent = 0.
        accBest = 0.
        best_model = None

        epochs_no_improve = 0
        early_stop = False

        while accCurrent < 0.99 and not early_stop:
            lossCurrent, accCurrent = self._PDCL(epoch, loader_tr, optimizer)

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

            for g in optimizer.param_groups:
                g['lr'] *= self.dlr # primal lr
            self.lr_dual = 0.05*self.dlr/(epoch**(1/16))  #dual lr non-summable diminishing  

        self.clf = best_model
