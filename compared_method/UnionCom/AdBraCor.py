import os
import sys
import time
import numpy as np
from Project import project_tsne, project_barycentric
from Match import match
from utils import *
from test import *
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
####################################################################
data_id = "AdBraCor"

####################################################################

class params():
    epoch_pd = 2000
    epoch_DNN = 200
    epsilon = 0.001
    lr = 0.001
    batch_size = 100
    rho = 10
    log_DNN = 50
    log_pd = 200
    manual_seed = 666
    delay = 0
    kmax = 20
    beta = 1
    col = []
    row = []
    output_dim = 32


def fit_transform(dataset, epoch_pd=2000, epoch_DNN=200,
                  epsilon=0.001, lr=0.001, batch_size=100, rho=10, beta=1,
                  log_DNN=50, log_pd=200, manual_seed=666, delay=0, kmax=40,
                  output_dim=32, distance='geodesic', project='tsne'):
    '''
    parameters:
    dataset: list of datasets to be integrated. [dataset1, dataset2, ...].
    epoch_pd: epoch of Prime-dual algorithm.
    epoch_DNN: epoch of training Deep Neural Network.
    epsilon: training rate of data matching matrix F.
    lr: training rate of DNN.
    batch_size: training batch size of DNN.
    beta: trade-off parameter of structure preserving and point matching.
    rho: training damping term.
    log_DNN: log step of training DNN.
    log_pd: log step of prime dual method
    manual_seed: random seed.
    distance: mode of distance, ['geodesic, euclidean'], default is geodesic.
    output_dim: output dimension of integrated data.
    project:　mode of project, ['tsne', 'barycentric'], default is tsne.
    ---------------------
    '''
    params.epoch_pd = epoch_pd
    params.epoch_DNN = epoch_DNN
    params.epsilon = epsilon
    params.lr = lr
    params.batch_size = batch_size
    params.rho = rho
    params.log_DNN = log_DNN
    params.log_pd = log_pd
    params.manual_seed = manual_seed
    params.delay = delay
    params.beta = beta
    params.kmax = kmax
    params.output_dim = output_dim

    time1 = time.time()
    init_random_seed(manual_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_num = len(dataset)

    row = []
    col = []
    dist = []
    kmin = []

    #### compute the distance matrix
    print("Shape of Raw data")
    for i in range(dataset_num):
        row.append(np.shape(dataset[i])[0])
        col.append(np.shape(dataset[i])[1])
        print("Dataset {}:".format(i), np.shape(dataset[i]))

        dataset[i] = (dataset[i] - np.min(dataset[i])) / (np.max(dataset[i]) - np.min(dataset[i]))

        if distance == 'geodesic':
            dist_tmp, k_tmp = geodesic_distances(dataset[i], params.kmax)
            dist.append(np.array(dist_tmp))
            kmin.append(k_tmp)

        if distance == 'euclidean':
            dist_tmp, k_tmp = euclidean_distances(dataset[i])
            dist.append(np.array(dist_tmp))
            kmin.append(k_tmp)

    params.row = row
    params.col = col

    # find correspondence between cells
    pairs_x = []
    pairs_y = []
    match_result = match(params, dataset, dist, device)
    for i in range(dataset_num - 1):
        cost = np.max(match_result[i]) - match_result[i]
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs_x.append(row_ind)
        pairs_y.append(col_ind)

    #  projection
    if project == 'tsne':

        P_joint = []
        for i in range(dataset_num):
            P_joint.append(p_joint(dist[i], kmin[i]))
        integrated_data = project_tsne(params, dataset, pairs_x, pairs_y, dist, P_joint, device)

    else:
        integrated_data = project_barycentric(dataset, match_result)

    print("---------------------------------")
    print("unionCom Done!")
    time2 = time.time()
    print('time:', time2 - time1, 'seconds')

    return integrated_data


def test_label_transfer_accuracy(integrated_data, datatype):
    test_UnionCom(integrated_data, datatype)


path = '../datasets/' + data_id
# data = np.load(os.path.join(path, 'rawdata.npy'), allow_pickle=True).item()
#
# data1 = data['exp'][0]
# data2 = data['exp'][1]
# type1 = data['type'][0]
# type2 = data['type'][1]

import anndata

X1 = anndata.read(os.path.join(path, "raw_data_rna.h5ad"))
X2 = anndata.read(os.path.join(path, "raw_data_atac.h5ad"))
data1 = X1.X
data2 = X2.X
type1 = X1.obs['cell_type']
type2 = X2.obs['cell_type']

# -------------------------------------------------------

# dist_tmp, k_tmp, not_connected, connect_element, index = Maximum_connected_subgraph(data3, params.kmax)
not_connected, connect_element, index = Maximum_connected_subgraph(data2, params.kmax)

if not_connected:
    data2 = data2[connect_element[index]]
    type2 = type2[connect_element[index]]

min_max_scaler = preprocessing.MinMaxScaler()
data2 = min_max_scaler.fit_transform(data2)
print(np.shape(data2))
# ------------------------------------------------------
# type1 = type1.astype(np.int)
# type2 = type2.astype(np.int)
# datatype = [type1,type2]
# inte = fit_transform([data1,data2])
# test_label_transfer_accuracy(inte, datatype)
# Visualize([data1,data2], inte, datatype, mode='PCA')

# type1 = type1.astype(int)
# type2 = type2.astype(int)

inte = fit_transform([data1, data2])

UnionCom_inte = dict({"inte": inte})

path = 'E:/experiment/scMGPF/results/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

np.save(os.path.join(path, 'UnionCom.npy'), UnionCom_inte)
