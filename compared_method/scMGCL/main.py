import os

import anndata
import numpy as np

from scMGCL import run
from evaluate_metrics import evaluate_model
import scanpy as sc

data_id = "BMMC_s1d1"
path = '../datasets/' + data_id
adata_rna = anndata.read_h5ad(os.path.join(path, "raw_data_rna.h5ad"))
adata_atac = anndata.read_h5ad(os.path.join(path, "raw_data_atac.h5ad"))
# load data
# adata_atac = anndata.read_h5ad('ATAC.h5ad')
# adata_rna = anndata.read_h5ad('RNA.h5ad')

adata_atac.obs['source'] = 'ATAC'
adata_rna.obs['source'] = 'RNA'


adata = anndata.concat([adata_rna, adata_atac], join='inner', merge='same')
adata.obs_names_make_unique()  # 确保观测名称唯一
adata.obs['cell_type'] = adata.obs['cell_type']

cell_type = 'cell_type'
adata.obs['cell_type'] = adata.obs[f'{cell_type}']

# the scMGCL function is called to return the trained adata directly
integrated = run(adata, adata_rna, adata_atac)

RNA = integrated[integrated.obs['source'] == 'RNA']
ATAC = integrated[integrated.obs['source'] == 'ATAC']

z_rna = RNA.obsm['integrated_embeddings']
z_atac = ATAC.obsm['integrated_embeddings']

inte = []
inte.append(z_rna)
inte.append(z_atac)

scmgcl_inte = dict({"inte": inte})

path = 'E:/experiment/DaOT/results/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

np.save(os.path.join(path, 'scMGCL.npy'), scmgcl_inte)

# rna_cell_types=RNA.obs['cell_type']
# atac_cell_types=ATAC.obs['cell_type']

# results = evaluate_model(z_rna, z_atac, rna_cell_types, atac_cell_types, integrated)
#
# # UMAP visualization
# sc.set_figure_params(dpi=400, fontsize=10)
# sc.pp.neighbors(integrated,use_rep='integrated_embeddings')
# sc.tl.umap(integrated,min_dist=0.1)
#
# sc.pl.umap(integrated, color=['source','cell_type'],title=['',''],wspace=0.3, legend_fontsize=10)
#
# # save results
# integrated.write('scMGCL_integrated.h5ad')


