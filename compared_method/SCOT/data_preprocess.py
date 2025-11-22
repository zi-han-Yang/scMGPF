#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 23:42:14 2021
@author: xiaokangyu
"""
from pandas import value_counts
import scanpy as sc
import scipy
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans,MiniBatchKMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sps
from collections import defaultdict


# def preprocess(data1, data2, cnames1, cnames2, gnames, metadata,
#                n_high_var=1000, hvg_list=None,
#                normalize_samples=True, target_sum=1e4, log_normalize=True,
#                normalize_features=True, pca_dim=100, scale_value=10.0,
#                use_reduction=True):
def preprocess(adata,
               n_high_var=1000, hvg_list=None,
               normalize_samples=True, target_sum=1e4, log_normalize=True,
               normalize_features=True, pca_dim=100, scale_value=10.0,
               use_reduction=True):
    """
    Preprocessing raw dataset
    Argument:
    ------------------------------------------------------------------
    - adata: `anndata.AnnData`, the annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells and columns to genes.

    - cluster_method: "str", the clustering algorithm to initizite celltype label["louvain","leiden","kmeans","minibatch-kmeans"]

    - resolution:'np.float', default:3.0, the resolution of louvain algorthm for scDML to initialize clustering

    - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.

    - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 1000, then the 1000 genes with the highest variance are designated as highly variable.

    - hvg_list: 'list',  a list of highly variable genes for seqRNA data

    - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.

    - target_sum: 'int',default 1e4,Total counts after cell normalization,you can choose 1e6 to do CPM normalization

    - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.

    - normalize_features: `bool`, If True, z-score normalize each gene's expression.

    - pca_dim: 'int', number of principal components

    - scale_value: parameter used in sc.pp.scale() which uses to truncate the outlier

    - num_cluster: "np.int", K parameter of kmeans

    Return:
    - normalized adata suitable for integration of scDML in following stage.
    """

    # cnames = np.hstack((cnames1, cnames2))
    # data_x = np.vstack([data1, data2])
    # adata = sc.AnnData(data_x)  # transposed, (gene, cell) -> (cell, gene)
    # adata.obs_names = cnames
    # adata.var_names = gnames
    # adata.obs = metadata.loc[cnames].copy()
    # nbatch = len(adata.obs["batch"].value_counts())

    normalized_adata = Normalization(adata, n_high_var, hvg_list,
                                     normalize_samples, target_sum, log_normalize,
                                     normalize_features, scale_value, use_reduction, pca_dim)

    # init_clustering(emb, resolution, cluster_method=cluster_method)

    # batch_index = normalized_adata.obs[batch_key].values
    # normalized_adata.obs["init_cluster"] = emb.obs["init_cluster"].values.copy()
    # num_init_cluster = len(emb.obs["init_cluster"].value_counts())

    return normalized_adata





def Normalization(adata, n_high_var = 1000,hvg_list=None,
                     normalize_samples = True,target_sum=1e4,log_normalize = True, 
                     normalize_features = True,scale_value=10.0,use_reduction=False, pca_dim=100):
    """
    Normalization of raw dataset 
    ------------------------------------------------------------------
    Argument:
        - adata: raw adata to be normalized

        - batch_key: `str`, string specifying the name of the column in the observation dataframe which identifies the batch of each cell. If this is left as None, then all cells are assumed to be from one batch.
    
        - n_high_var: `int`, integer specifying the number of genes to be idntified as highly variable. E.g. if n_high_var = 1000, then the 1000 genes with the highest variance are designated as highly variable.
       
        - hvg_list: 'list',  a list of highly variable genes for seqRNA data
        
        - normalize_samples: `bool`, If True, normalize expression of each gene in each cell by the sum of expression counts in that cell.
        
        - target_sum: 'int',default 1e4,Total counts after cell normalization,you can choose 1e6 to do CPM normalization
            
        - log_normalize: `bool`, If True, log transform expression. I.e., compute log(expression + 1) for each gene, cell expression count.
        
        - normalize_features: `bool`, If True, z-score normalize each gene's expression.

    Return:
        Normalized adata
    ------------------------------------------------------------------
    """
    
    n, p = adata.shape
    
    if(normalize_samples):
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if(log_normalize):
        sc.pp.log1p(adata)
    
    if hvg_list is None:
        sc.pp.highly_variable_genes(adata,n_top_genes=n_high_var,subset=True)
    else:
        adata = adata[:, hvg_list]

    if normalize_features:
        adata_sep=[]
        for batch in np.unique(adata.obs["batch"]):
            sep_batch=adata[adata.obs["batch"]==batch]
            sc.pp.scale(sep_batch,max_value=scale_value)
            if use_reduction:
                sep_batch = dimension_reduction(sep_batch, pca_dim)
            adata_sep.append(sep_batch)

        # adata=sc.AnnData.concatenate(*adata_sep)

    #adata.layers["normalized_input"] = adata.X        
    return adata_sep
  
def dimension_reduction(adata,dim=100,verbose=True,log=None):
    """
    apply dimension reduction with normalized dataset 
    ------------------------------------------------------------------
    Argument:
        - adata: normazlied adata  

        - dim: ’int‘, default:100, number of principal components in PCA dimension reduction
    
        - verbose: print additional infomation

    Return:
        diemension reudced adata
    ------------------------------------------------------------------
    """
    # if(verbose):
    #     log.info("Calculate PCA(n_comps={})".format(dim))
        
    if(adata.shape[0]>300000):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    sc.tl.pca(adata,n_comps=dim)
    emb=sc.AnnData(adata.obsm["X_pca"])
    return emb
    
def init_clustering(emb,reso=3.0,cluster_method="louvain",num_cluster=50):
    """
    apply clustering algorithm in PCA embedding space, defualt: louvain clustering
    ------------------------------------------------------------------
    Argument:
        - emb: 'AnnData',embedding data of adata(PCA)

        - reso: ’float‘, default:3.0, resolution defined in louvain(or leiden) algorithm
        
        - cluster_method: 'str', clustering algorothm to initize scDML cluster
        
        - num_cluster: 'int', default:40, parameters for kmeans(or minibatch-kmeans) clustering algorithm
    ------------------------------------------------------------------
    """
    if(cluster_method=="louvain"):
        sc.pp.neighbors(emb,random_state=0)
        sc.tl.louvain(emb,resolution=reso,key_added="init_cluster")

    elif(cluster_method=="leiden"):
        sc.pp.neighbors(emb,random_state=0)
        sc.tl.leiden(emb,resolution=reso,key_added="init_cluster")

    elif(cluster_method=="kmeans"):
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(emb.X) 
        emb.obs['init_cluster'] = kmeans.labels_.astype(str)
        emb.obs['init_cluster'] = emb.obs['init_cluster'].astype("category")   

    elif(cluster_method=="minibatch-kmeans"): # this cluster method will reduce time and memory but less accuracy
        kmeans = MiniBatchKMeans(init='k-means++',n_clusters=num_cluster,random_state=0,batch_size=64).fit(emb.X)
        emb.obs['init_cluster'] = kmeans.labels_.astype(str)
        emb.obs['init_cluster'] = emb.obs['init_cluster'].astype("category")

    else:
        raise IOError
