import numpy as np
import pandas as pd
import anndata
import os

from scTopoGAN_Functions import get_TopoAE_Embeddings, run_scTopoGAN

# load data
# RNA = pd.read_csv('Data/PBMC Multiome/RNA_PCA.csv',header=0,index_col=0)
# ATAC = pd.read_csv('Data/PBMC Multiome/ATAC_LSI.csv',header=0,index_col=0)
data_id = 'CITE_PBMC'
# dataset_dir = '../datasets/' + data_id
# data = np.load(os.path.join(dataset_dir, "rawdata.npy"), allow_pickle=True)
# RNA = data[0]
# ATAC = data[1]

path = '../datasets/' + data_id
RNA = anndata.read_h5ad(os.path.join(path, "raw_data_rna.h5ad"))
ATAC = anndata.read_h5ad(os.path.join(path, "raw_data_atac.h5ad"))

# RNA = np.load('/Users/guoyin/Desktop/UZH/contrastive_learning/method/BC6/data/PBMC1/PBMC1_data2.npy', allow_pickle=True)
# ATAC = np.load('/Users/guoyin/Desktop/UZH/contrastive_learning/method/BC6/data/PBMC1/PBMC1_data3.npy', allow_pickle=True)


# Step 1: Get TopoAE embeddings
# set topology_regulariser_coefficient between 0.5 to 3.0

target_latent = get_TopoAE_Embeddings(Manifold_Data=RNA, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC",
                                      AE_arch=[50, 32, 32, 8], topology_regulariser_coefficient=3, initial_LR=0.001)

source_latent = get_TopoAE_Embeddings(Manifold_Data=ATAC, batch_size=50, autoencoder_model="MLPAutoencoder_PBMC",
                                      AE_arch=[50, 32, 32, 8], topology_regulariser_coefficient=0.5, initial_LR=0.001)

## Step 2: Manifold alignment using scTopoGAN
source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ATAC", target_tech_name="RNA",
                               batch_size=512, topology_batch_size=1000, total_epochs=1001, num_iterations=15,
                               checkpoint_epoch=100, g_learning_rate=1e-4, d_learning_rate=1e-4, path_prefix="Results")
# source_aligned = run_scTopoGAN(source_latent, target_latent, source_tech_name="ATAC", target_tech_name="RNA",
#                                batch_size=512, topology_batch_size=1000, total_epochs=1, num_iterations=1,
#                                checkpoint_epoch=1, g_learning_rate=1e-4, d_learning_rate=1e-4, path_prefix="Results")

inte = []
inte.append(target_latent.to_numpy())
inte.append(source_aligned.to_numpy())

scTopoGAN_inte = dict({"inte": inte})

path = 'E:/experiment/scMGPF/results/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

np.save(os.path.join(path, 'scTopoGAN.npy'), scTopoGAN_inte)

# source_aligned.to_csv('ATAC_aligned.csv')
