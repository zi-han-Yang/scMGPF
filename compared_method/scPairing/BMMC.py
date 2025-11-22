#!/usr/bin/env python
# coding: utf-8

import anndata as ad
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import scanpy as sc
import sklearn
from scPairing import scPairing
import os
import anndata
import warnings

warnings.filterwarnings("ignore")
data_id = "BMMC_s1d1"

path = '../datasets/' + data_id
data1 = anndata.read(os.path.join(path, "raw_data_rna.h5ad"))
data2 = anndata.read(os.path.join(path, "raw_data_atac.h5ad"))
data1.layers['counts'] = data1.X.copy()
data2.layers['counts'] = data2.X.copy()

data2.layers['binary'] = sklearn.preprocessing.binarize(data2.X)
data1.obsm["X_pca"] = data1.X
data2.obsm["X_lsi"] = data2.X
model = scPairing(
    data1,
    data2,
    counts_layer=['counts', 'binary'],
    transformed_obsm=['X_pca', 'X_lsi'],
    use_decoder=False,
    seed=0
)
model.train(epochs=200)
latents = model.get_latent_representation()
data1.obsm['mod1_features'] = data2.obsm['mod1_features'] = latents[0]
data1.obsm['mod2_features'] = data2.obsm['mod2_features'] = latents[1]


inte = [data1.obsm['mod1_features'], data2.obsm['mod2_features']]

scPairing_inte = dict({"inte": inte})

result_dir = 'E:/experiment/scMGPF/results/' + data_id
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

np.save(os.path.join(result_dir, 'scPairing.npy'), scPairing_inte)
