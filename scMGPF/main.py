import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from utils import *
from model import scMGPF, scMGPF_learning, get_embedding
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
if not os.path.exists('../logs/'):
    os.makedirs('../logs/')
if not os.path.exists('../results/'):
    os.makedirs('../results/')

# dataset_name = "P0BraCor"
# dataset_name = "CITE_PBMC"
# dataset_name = "Tea_PBMC"
# dataset_name = "AdBraCor"
dataset_name = "P0BraCor"

config_args = load_config('../config/' + dataset_name)
args = argparse.Namespace(**config_args)
path = "E:/experiment/scMGPF/"
dataset_dir = path + 'data/' + dataset_name
result_dir = path + "results/" + dataset_name
eva_dir = path + "eva/" + dataset_name
vis_dir = path + 'vis/' + dataset_name
model_path = path + 'logs/model_' + dataset_name


def main():
    print('start')
    rna_adata = sc.read_h5ad(os.path.join(dataset_dir, "raw_data_rna.h5ad"))
    atac_adata = sc.read_h5ad(os.path.join(dataset_dir, "raw_data_atac.h5ad"))
    cell_num = [rna_adata.shape[0], atac_adata.shape[0]]

    rna = torch.tensor(rna_adata.X, dtype=torch.float32).to(device)
    atac = torch.tensor(atac_adata.X, dtype=torch.float32).to(device)

    rna_clusters_label = torch.tensor(rna_adata.obs['clusters'].to_numpy(), dtype=torch.long, device=device)
    atac_clusters_label = torch.tensor(atac_adata.obs['clusters'].to_numpy(), dtype=torch.long, device=device)
    clusters_label = [rna_clusters_label, atac_clusters_label]
    print(rna_clusters_label.shape)

    data = [rna, atac]
    get_graph = Construct_Graph_and_intra_distances(data, cell_num, device)
    get_graph.construct_knn_graph(k=10, graph_mode="distance", metric="minkowski")
    rna_graph = get_graph.graphs[0]
    atac_graph = get_graph.graphs[1]

    gene_dim = rna.shape[1]
    print(len(rna.shape))
    peak_dim = atac.shape[1]

    model = scMGPF(
        [rna, atac], gene_dim=gene_dim, peak_dim=peak_dim,
        hidden_dim=256, latent_dim=32, cell_num=cell_num, dropout=0.01
    ).to(device)

    trainer = scMGPF_learning(model, rna, atac, 10, args.k_clusters,
                              learning_rate=0.003, weight_decay=1e-5, num_epochs=300, seed=666,
                              model_path=model_path, clusters_label=clusters_label, device=device)
    trainer.train_model()

    get_embedding(data, rna_graph, atac_graph, dataset_name, model_path, model, cell_num, device)

    pred_labels = np.loadtxt("result/rna_Classifier_label_predict.txt")
    print(f"Predicted labels shape: {pred_labels.shape}, unique: {np.unique(pred_labels)}")

    if isinstance(clusters_label[0], torch.Tensor):
        true_labels = clusters_label[0].cpu().numpy()
    else:
        true_labels = clusters_label[0]
    print(f"True labels shape: {true_labels.shape}, unique: {np.unique(true_labels)}")

    correct = np.sum(pred_labels == true_labels)
    accuracy = correct / len(true_labels)

    print(f"LTA: {accuracy:.4f}")

    cm = confusion_matrix(true_labels, pred_labels)
    print(f"Confusion Matrix Shape: {cm.shape}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix of Cell Type Prediction')
    plt.show()


if __name__ == "__main__":
    main()
