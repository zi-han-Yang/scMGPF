import random
import time
import yaml
import json
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from torch import block_diag
from scipy.sparse import diags, coo_matrix, block_diag
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, degree

# For computing graph distances:
from sklearn.neighbors import kneighbors_graph


def init_random_seed(manual_seed):
    if manual_seed is None:
        seed = int(time.time() * 1000) % (2 ** 32)  #Generate seeds based on time
    else:
        if not isinstance(manual_seed, int) or manual_seed < 0:
            raise ValueError("manual_seed must be a non-negative integer.")
        seed = manual_seed

    print(f"Using random seed: {seed}")
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def load_config(config_name, verbose=True):
    if '.yaml' not in config_name:
        config_name += '.yaml'
    with open(config_name, 'r') as f:
        f_str = f.read()
        dic = yaml.safe_load(f_str)
        if verbose:
            js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
            print(js)
        return dic


def check_balance(data, cell_ratio_thresholds=1.1, feature_ratio_thresholds=2.0):
    """
    Evaluate the balance of RNA and ATAC data, including the number of cells and features.
        Parameter
            cell_balance_status: Cell balance status, default true
            feature_balance_status: Feature balance status, default true
            cell_ratio_thresholds: Cell quantity ratio threshold, default 1.1.
            feature_ratio_thresholds: Threshold for the ratio of feature quantities, default 2.0.
        Return
            dict: It includes imbalanced indicators and other indicators
    """
    if len(data) < 2:
        raise ValueError("At least two datasets  are required.")

    result = {
        'cell_details': {},
        'feature_details': {},
        'cell_balance_status': True,
        'feature_balance_status': True,
    }
    # Cell Quantity analysis
    cell_counts = [data.shape[0] for data in data]
    cell_min = min(cell_counts)
    cell_ratio = max(cell_counts) / cell_min if cell_min > 0 else float('inf')

    result['cell_details'] = {
        'rna_cells': cell_counts[0],
        'atac_cells': cell_counts[1],
        'ratio': cell_ratio
    }

    result['cell_balance_status'] = cell_ratio <= cell_ratio_thresholds
    # Feature Quantity Analysis
    feature_counts = [data.shape[1] for data in data]
    feature_min = min(feature_counts)
    feature_ratio = max(feature_counts) / feature_min if feature_min > 0 else float('inf')

    result['feature_details'] = {
        'rna_features': feature_counts[0],
        'atac_features': feature_counts[1],
        'ratio': feature_ratio
    }

    result['feature_balance_status'] = feature_ratio <= feature_ratio_thresholds

    return result


class Construct_Graph_and_intra_distances:

    def __init__(self, data, cell_num, device="cuda"):
        self.data = data  
        self.device = device
        self.graphs = []  
        self.disgraph_x = None  # intra-datasets graph distances for datasets 1 (X)
        self.disgraph_y = None  # intra-datasets graph distances for datasets 2 (y)
        self.intra_graphDists = []  # Holds intra-domain graph distances for each input dataset
        self.cell_num = cell_num

    def construct_knn_graph(self, graph_mode, k, metric="minkowski", p=2, return_edge_index=True):
        """
        Parameter
            k (int): Number of neighbors.
            mode (str): "connectivity" (binary adjacency matrix) or "distance" (weighted adjacency matrix).
            metric (str): Distance measurement, with options including "minkowski" (default p=2 for Euclidean distance), "jaccard", and "correlation".
            p (float): The exponent of the minkowski distance, valid only when metric="minkowski".
            return_edge_index (bool): Whether to return the edge_index format of PyTorch Geometric.
        Return
            list: KNN graphs of each dataset, where graphs[0] are tuples (edge indexes or sparse matrices).
        """
        assert graph_mode in ["connectivity", "distance"], "Mode argument must be 'connectivity' or 'distance'."
        assert metric in ["minkowski", "jaccard",
                          "correlation"], "Metric must be 'minkowski', 'jaccard', or 'correlation'."
        idx = 0 if metric == "minkowski" else 1
        if len(self.data[idx].shape) != 2:
            raise ValueError(f"Data is not a 2D matrix.")

        for i, data in enumerate(self.data):
            if isinstance(data, torch.Tensor):
                X = data.cpu().numpy()  
            else:
                X = np.asarray(data)  
                
                # Convert the data for Jaccard distance to Boolean type
            if metric == "jaccard" and X.dtype != np.bool_:
                X = (X != 0).astype(np.bool_)  # Convert to Boolean type to adapt to Jaccard metrics

            # Initialize  NearestNeighbors
            if metric == "minkowski":

                nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, p=p) # Excluding its own data points, n_neighbors=k + 1 includes its own data points

            else:
                nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
            # Calculate the KNN graph
            nbrs.fit(X)
            if graph_mode == "distance":
                distances, indices = nbrs.kneighbors(X)

                if indices.shape[1] != distances.shape[1]:
                    raise ValueError(f"Edge index and weight mismatch for data")
                # Build a weighted adjacency matrix
                rows = np.repeat(np.arange(X.shape[0]), k)  
                indices = indices[:, 1:]
                distances = distances[:, 1:]  
                cols = indices.flatten()  
                distances = distances.flatten()
                if return_edge_index:

                    edge_index_np = np.array([rows, cols], dtype=np.int64)
                    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
                    # edge_index = to_undirected(edge_index, num_nodes=X.shape[0])
                    edge_weight = torch.tensor(distances, dtype=torch.float, device=self.device)
                    self.graphs.append((edge_index, edge_weight))
                else:
                    graph = sp.csr_matrix((distances, (rows, cols)), shape=(X.shape[0], X.shape[0])) 
                    self.graphs.append(graph)

            else:  # mode == "connectivity"
                graph = nbrs.kneighbors_graph(X, mode="connectivity")  
                if return_edge_index:
                    rows, cols = graph.nonzero()
                    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
                    edge_index = to_undirected(edge_index)
                    self.graphs.append((edge_index, None)) 
                else:
                    self.graphs.append(graph)
        return self.graphs

    def graph_distances_matrix_optimized(self):
        for i, graph_tuple in enumerate(self.graphs):
            if isinstance(graph_tuple, tuple): 
                rows, cols = graph_tuple[0].cpu().numpy()
                weights = graph_tuple[1].cpu().numpy()
                num_nodes = self.cell_num[i]
                graph = csr_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
            else:
                graph = graph_tuple
            # Efficiently calculate the shortest path between all node pairs
            dist_matrix = dijkstra(csgraph=graph, directed=False)
            # Handle infinite values generated by disconnected components
            inf_val = np.max(dist_matrix[np.isfinite(dist_matrix)])
            dist_matrix[np.isinf(dist_matrix)] = inf_val
            # Normalize and convert to Tensor
            dist_matrix /= dist_matrix.max()
            np.fill_diagonal(dist_matrix, 0)

            self.intra_graphDists.append(torch.tensor(dist_matrix, dtype=torch.float, device=self.device))

        return self.intra_graphDists

    def graph_distances_matrix(self):

        for i, graph in enumerate(self.graphs[0:1]):
            if not isinstance(graph, tuple) or len(graph) != 2:
                raise ValueError(f"Graph[{i}] must be a tuple of (edge_index, edge_weight).")
            edge_index, edge_weight = graph
            if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_weight, torch.Tensor):
                raise ValueError(f"edge_index and edge_weight for Graph[{i}] must be torch.Tensor.")

            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)

            if edge_index.shape[1] == 0 or self.cell_num == 0:
                raise ValueError(f"Graph[{i}] is empty or has no nodes.")
            # Initialize the distance matrix (infinity indicates unconnected)
            inf = float('inf')
            dist_matrix = torch.full((self.cell_num[i], self.cell_num[i]), inf, device=self.device)
            dist_matrix[torch.arange(self.cell_num[i]), torch.arange(self.cell_num[i])] = 0
            # Fill edge weights
            row, col = edge_index
            dist_matrix[row, col] = edge_weight
            dist_matrix[col, row] = edge_weight  # Ensure an undirected graph

            # Floyd-Warshall
            for k in range(self.cell_num[i]):
                dist_matrix = torch.min(
                    dist_matrix,
                    dist_matrix[:, k].unsqueeze(1) + dist_matrix[k, :].unsqueeze(0)
                )

            max_dist = torch.max(dist_matrix[dist_matrix != inf])
            if torch.isnan(max_dist) or max_dist == 0:
                max_dist = torch.tensor(1.0, device=self.device)  # Avoid division by zero

            dist_matrix[dist_matrix == inf] = max_dist

            dist_matrix = dist_matrix / (dist_matrix.max() + 1e-10)
            dist_matrix.fill_diagonal_(0)  # Make sure the diagonal is 0

            self.intra_graphDists.append(dist_matrix)

        return self.intra_graphDists


def get_spatial_distance_matrix(data, metric="euclidean"):
    Cdata = sp.spatial.distance.cdist(data, data, metric=metric)
    return Cdata / Cdata.max()


def get_marginals(data, marginals_mode="uniform", knn_edge_indices=None, metadata=None, normalization=True,
                  device="cuda"):
    """
    Parameter
        mode (str): Distribution type, options:
        - "uniform": Evenly distributed (1/num_cells).
        - "degree": Based on the degree of the KNN graph, the weight is proportional to the number of neighbors.
        - "expression": Based on total expression level (RNA) or peak count (ATAC).
        - "metadata": Based on metadata (such as cell type weights).
        knn_edge_indices (list, optional): KNN graph edge index list, used in degree mode.
        normalization (bool): Whether to normalize the distribution (with a sum of 1).
    Return
        list: Edge distribution list, with each element being a tensor of shape (num_cells). (n,0) and (m,0)
    """
    if data is None:
        raise ValueError("self.data is empty.")

    marginals = []

    for i, data in enumerate(data):
        if len(data.shape) != 2:
            raise ValueError(f"Data[{i}] is not a 2D matrix.")
        num_cells = data.shape[0]
        if num_cells <= 0:
            raise ValueError(f"Data[{i}] has invalid number of cells.")

        if marginals_mode == "uniform":

            marginal_dist = torch.ones(num_cells, device=device) / num_cells

        elif marginals_mode == "degree":

            if knn_edge_indices is None or i >= len(knn_edge_indices):
                raise ValueError(f"KNN edge index for data[{i}] is required for degree mode.")
            edge_index = knn_edge_indices[i].to(device)
            deg = degree(edge_index[i][0], num_nodes=num_cells, dtype=torch.float)
            marginal_dist = deg / (deg.sum() + 1e-8)  # Normalizing Degree

        elif marginals_mode == "expression":
           # Based on the differences in total expression level (RNA) or peak count (ATAC) activity

            expr_sum = data.sum(dim=1)  # Total characteristic value of each cell

            marginal_dist = expr_sum / (expr_sum.sum() + 1e-8)  # Normalization

        elif marginals_mode == "metadata":
            # Based on metadata (such as cell type weights)
            if metadata is None or i >= len(metadata):
                raise ValueError(f"Metadata for data[{i}] is required for metadata mode.")
            # Suppose metadata[i] is the weight of cell types (for example, rare cell types have a higher weight)
            weights = torch.tensor(metadata[i], dtype=torch.float, device=device)
            marginal_dist = weights / (weights.sum() + 1e-8)

        else:
            raise ValueError(
                f"Unsupported mode: {marginals_mode}. Choose from 'uniform', 'degree', 'expression', 'metadata'.")
        # Ensure non-negative and normalized

        if normalization:
            marginal_dist = torch.clamp(marginal_dist, min=0)
            marginal_dist = marginal_dist / (marginal_dist.sum() + 1e-8)

        marginals.append(marginal_dist)

    return marginals


