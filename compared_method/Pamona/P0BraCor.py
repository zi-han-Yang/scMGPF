#!/usr/bin/env python
# coding: utf-8


from Pamona import Pamona
import numpy as np
import os
import anndata
import warnings
warnings.filterwarnings("ignore")
data_id = "P0BraCor"

path = '../datasets/' + data_id
data1 = anndata.read(os.path.join(path, "raw_data_rna.h5ad"))
data2 = anndata.read(os.path.join(path, "raw_data_atac.h5ad"))
X1 = data1.X
X2 = data2.X
data = [X1, X2]

# path = '../datasets/' + data_id
# data = np.load(os.path.join(path, 'rawdata.npy'), allow_pickle=True).item()
# data1 = data['exp'][0]
# data2 = data['exp'][1]
# data = [data1, data2]


Pa = Pamona(n_neighbors=10, Lambda=10)

integrated_data, T = Pa.run_Pamona(data)

Pamona_inte = dict({"inte": integrated_data})

result_dir = 'E:/experiment/scMGPF/results/' + data_id
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

np.save(os.path.join(result_dir, 'Pamona.npy'), Pamona_inte)
