from sklearn.cluster import KMeans
import sys
import argparse

# from util import load_config
from vis.plot_func import *
from eva.load_result import *
from eva.eval import *
import warnings
warnings.filterwarnings("ignore")

dataset_name = "P0BraCor"

dataset_type = 'RNA_ATAC'
GAM_name = 'Signac'

path = "E:/experiment/scMGPF/"
dataset_dir = path + 'data/'
result_dir = path + "results/" + dataset_name
eva_dir = path + "eva/" + dataset_name
vis_dir = path + 'vis/' + dataset_name

methods = ['raw_data', 'JointMDS', 'MMDMA', 'Pamona', 'SCOT', 'UnionCom', 'scTopoGAN', 'scMGCL', 'scPairing', 'scMGPF']

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
if not os.path.exists(eva_dir):
    os.makedirs(eva_dir)

adata, anno_rna, anno_atac, clu_rna, clu_atac = load_result(dataset_name, dataset_dir, result_dir,
                                                            methods)
adata.obs_names_make_unique()  
obsm_names = adata.obsm_keys()

n_clusters = len(np.unique(adata.obs['cluster']))
cell_types = np.unique(adata.obs['cell_type'])


#       1.Plot umap visualization ##############################

omic_colors = ['#60A5C0', '#F75B34']
cell_type_colors_all = [
    "#FF0000", "#CF3907", "#DB538E", "#fca2a0",
    "#FA6540", "#F7A308", "#FFFF00", '#958431', "#723809FF", "#6D6B6B",
    "#EA00FF", "#971C97", "#7D40F7", "#0000FF", "#0288D1",
    "#b1c6e4", "#00FCDA", "#89DF86", "#00FF00", "#199185", "#2C9B2C93",
    "#161515", "#DE0A58"]
cell_type_colors = cell_type_colors_all[:n_clusters]

# umap_plot(adata, omic_colors, cell_type_colors, vis_dir)
# umap_clusters_plot(adata, cell_type_colors, eva_dir, vis_dir)

#      2.Plot PAGA graph #########################################

paga_plot(adata, obsm_names, cell_type_colors, vis_dir)

#      3.Plot metrics  graph ################################

eva_metrics = pd.read_csv(eva_dir + '/eva_metrics_' + dataset_name + '.csv', index_col=0, header=0)

obsm_names_metrics = []

for i, method in enumerate(list(eva_metrics.index)):
    if len(method.split('_')) == 2 and method != 'raw_data':
        obsm_names_metrics.append(method.split('_')[0])
    else:
        obsm_names_metrics.append(method)

colors = [
    "#5f6f85", "#FF7F00", "#DB7093", "#FFD000", "#68ECC2",
    "#A65628", "#08CC08", "#da13f5", "#1E90FF", "#E41A1C",
    "#ff00dd", "#984EA3",  "#999999"]
method_colors = {}
for i in range(len(obsm_names_metrics)):
    method = obsm_names_metrics[i]
    color = colors[i]
    method_colors[method] = color

# metrics_plot(eva_metrics, method_colors, vis_dir)

#      4.Plot scatter graph #########################################

# scatter_plot(eva_metrics, method_colors, vis_dir)

#     5.Plot Radar graph  ##################################################

# radar_plot(eva_metrics, vis_dir, method_colors)

#    6.Plot acc confusion matrix graph####################################

accuracy = {}
CM = {}  
cluster_labels_np = np.concatenate([clu_rna, clu_atac], axis=0) 
for i, method in enumerate(obsm_names):
    acc, heatmap_matrix = get_accuracy_score(adata.obsm[method], cluster_labels_np, n_clusters, eva_dir, method)
    accuracy[method] = acc
    CM[method] = heatmap_matrix

# clu_heatmap_plot(adata, obsm_names, CM, accuracy, vis_dir, keys='clu_heatmap_plot')

#      7.Plot total performance graph #####################################

# total_performance_plot(eva_metrics, method_colors, vis_dir)

#      8.Plot classified_confusion_matrices  graph #####################################

# classified_confusion_matrices(adata, eva_dir, vis_dir, clu_rna)

