from data_preprocess import *
import pandas as pd
import anndata
from sklearn.decomposition import TruncatedSVD
import os
import sys
import scanpy as sc

current_directory = os.path.dirname(os.path.abspath(__file__))

data_id = "Tea_PBMC"

n_high_var = 2000
use_reduction = True

dataset_dir = "../../data/Tea_PBMC"
RNA_data = anndata.read(os.path.join(dataset_dir, 'GASM/rna.h5ad'))
ATAC_data = anndata.read(os.path.join(dataset_dir, 'GASM/atac.h5ad'))

RNA_data.obs['batch'] = 'RNA_c'
ATAC_data.obs['batch'] = 'ATAC_c'

pca_rna_data,  rna_adata = preprocess_rna(RNA_data, n_components=50)
lsa_atac_data, atac_adata = preprocess_atac(ATAC_data, n_components=50)


data1 = sc.AnnData(RNA_data.obsm['X_pca'])
data2 = sc.AnnData(ATAC_data.obsm['X_lsi'])
data1.obs_names = RNA_data.obs_names
data2.obs_names = ATAC_data.obs_names
data1.obs['clusters'] = RNA_data.obs['clusters']
data2.obs['clusters'] = ATAC_data.obs['clusters']
data1.obs['cell_type'] = RNA_data.obs['cell_type']
data2.obs['cell_type'] = ATAC_data.obs['cell_type']

path = '../datasets/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

data1.write_h5ad(os.path.join(path, "raw_data_rna.h5ad"))
data2.write_h5ad(os.path.join(path, "raw_data_atac.h5ad"))
