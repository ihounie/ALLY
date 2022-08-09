# Snippet to decide how many num workers to use

from time import time
import multiprocessing as mp
import dataset
from torch.utils.data import DataLoader


X_tr, Y_tr, X_te, Y_te = dataset.get_dataset("SVHN", "./data/")

for num_workers in range(1, 20, 1):  
    train_loader = DataLoader(X_tr, shuffle=True ,num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))