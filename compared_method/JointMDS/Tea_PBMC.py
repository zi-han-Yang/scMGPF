import os
import torch
import argparse
import numpy as np

from joint_mds import JointMDS
from scipy.spatial.distance import pdist
import utils.scores as scores
from utils.utils import plot_embedding, geodesic_dist
import anndata
import warnings
warnings.filterwarnings("ignore")
####################################################################
ncomp = 2
neighbor = 10
data_id = "Tea_PBMC"
####################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"
print("Running on", device)


def main():
    parser = argparse.ArgumentParser(
        description="Joint MDS for solving protein structure alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--components", type=int, default=ncomp, help="number of components"
    )

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    path = '../datasets/' + data_id
    data1 = anndata.read_h5ad(os.path.join(path, "raw_data_rna.h5ad"))
    data2 = anndata.read_h5ad(os.path.join(path, "raw_data_atac.h5ad"))
    X1 = data1.X
    X2 = data2.X

    # X1 = torch.from_numpy(data1.X).to(device)
    # X2 = torch.from_numpy(data2.X).to(device)

    # data = np.load(os.path.join(path, 'rawdata.npy'), allow_pickle=True).item()
    # X1 = data['exp'][0]
    # X2 = data['exp'][1]

    print(X1.shape)
    print(X2.shape)

    D1 = geodesic_dist(X1, k=neighbor, mode="connectivity", metric="correlation")
    D2 = geodesic_dist(X2, k=neighbor, mode="connectivity", metric="correlation")
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    JMDS = JointMDS(
        n_components=args.components,
        alpha=0.3,
        eps=0.1,
        max_iter=300,
        eps_annealing=False,
        dissimilarity="precomputed",
    )
    Z1, Z2, P = JMDS.fit_transform(D1, D2)

    Z1, Z2 = Z1.numpy(), Z2.numpy()

    inte = [Z1, Z2]

    JointMDS_inte = dict({"inte": inte})

    path = 'E:/experiment/scMGPF/results/' + data_id
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, 'JointMDS.npy'), JointMDS_inte)


if __name__ == "__main__":
    main()
