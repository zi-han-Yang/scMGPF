#!/usr/bin/env python
# coding: utf-8

import sys
import src.utils as ut
import src.evals as evals
import src.scot2 as sc
import numpy as np
import os
import anndata



# In[2]:

####################################################################
numk = 10
data_id = "P0BraCor"
####################################################################


# path = '../datasets/' + data_id
# data = np.load(os.path.join(path, 'rawdata.npy'), allow_pickle=True).item()
# X = data['exp'][0]
# y = data['exp'][1]

path = '../datasets/' + data_id
data1 = anndata.read(os.path.join(path, "raw_data_rna.h5ad"))
data2 = anndata.read(os.path.join(path, "raw_data_atac.h5ad"))
X = data1.X
y = data2.X

# initialize SCOT object
scot = sc.SCOT(X, y)
# call the alignment with z-score normalization
X_aligned, y_aligned = scot.align(k=numk, e=5e-3,  normalize=True, norm="l2")

inte = []
inte.append(X_aligned)
inte.append(y_aligned[0])

SCOT_inte = dict({"inte": inte})

path = 'E:/experiment/scMGPF/results/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

np.save(os.path.join(path, 'SCOT.npy'), SCOT_inte)
