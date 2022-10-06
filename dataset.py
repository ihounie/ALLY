import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import os
from torchvision import transforms as T

def get_dataset(name, path, redund = 0):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'STL10':
        return get_STL10(path, redund)
    elif name == 'SVHN':
        return get_SVHN(path, redund)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'TINY_IMAGENET':
        return get_TINY_IMAGENET(path)
    else:
        assert 0, "Wrong Dataset Name."

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    X_tr = X_tr
    Y_tr = Y_tr
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path, redund = 0):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = data_tr.labels
    X_te = data_te.data
    Y_te = data_te.labels
    X_tr = X_tr
    Y_tr = Y_tr

    for i in range(redund):
        X_tr = np.concatenate((X_tr, X_tr), axis = 0)
        Y_tr =  np.concatenate((Y_tr, Y_tr), axis = 0) 
        
    Y_tr = torch.from_numpy( Y_tr )
    Y_te = torch.from_numpy( Y_te )

    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_STL10(path, redund = 0):
    def read_all_images(path_to_data):

        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)

            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    Y_tr = read_labels('./data/STL10/stl10_binary/train_y.bin')
    X_tr = read_all_images('./data/STL10/stl10_binary/train_X.bin')

    Y_te = read_labels('./data/STL10/stl10_binary/test_y.bin')
    X_te = read_all_images('./data/STL10/stl10_binary/test_X.bin')
    X_tr = np.transpose(X_tr, (0,3,1,2))
    X_te = np.transpose(X_te, (0,3,1,2))
    Y_tr -= 1 #originally 1 to 10
    Y_te -= 1 
    id_s = int(len(X_te)*0.75)
    #Just keeping 2000 samples for testing 
    X_tr = np.concatenate((X_tr, X_te[:id_s]), axis = 0)
    Y_tr = np.concatenate((Y_tr, Y_te[:id_s]), axis = 0) 
    X_te =  X_te[id_s:]
    Y_te = Y_te[id_s:] 

    for i in range(redund):
        X_tr = np.concatenate((X_tr, X_tr), axis = 0)
        Y_tr =  np.concatenate((Y_tr, Y_tr), axis = 0) 

    Y_tr = torch.from_numpy( Y_tr )
    Y_te = torch.from_numpy( Y_te )
    return X_tr, Y_tr, X_te, Y_te

def get_TINY_IMAGENET(path):

    # Define main data directory
    DATA_DIR = './data/tiny-imagenet-200' 
    # Handle training data
    data_tr = datasets.ImageFolder(DATA_DIR+'/train/',  transform=None)
    data_ld = DataLoader(data_tr, batch_size = 20000, shuffle = True)

    X_tr, Y_tr = next(iter(data_ld))
    X_tr = X_tr.numpy()
    Y_tr = Y_tr.numpy()    

    data_te = datasets.ImageFolder(DATA_DIR+'/val/images',  transform=None)
    data_ld = DataLoader(data_te, batch_size = 4000, shuffle = True)

    X_te, Y_te = next(iter(data_ld))   
    X_te = X_te.numpy()
    Y_te = Y_te.numpy()   
    
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name in ['SVHN', 'STL10' ]:
        return DataHandler2
    elif name in ['CIFAR10', 'MNIST']:
        return DataHandler3
    else:
        return DataHandler4

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
