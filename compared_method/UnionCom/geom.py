import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.sparse as sp
from itertools import chain
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def geodesic_distances(X, kmax):
	kmin = 5
	nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
	knn = nbrs.kneighbors_graph(X, mode='distance')
	connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
	while connected_components is not 1:
		if kmin > np.max((kmax, 0.01*len(X))):
			break
		kmin += 2
		nbrs = NearestNeighbors(n_neighbors=kmin, metric='euclidean', n_jobs=-1).fit(X)
		knn = nbrs.kneighbors_graph(X, mode='distance')
		connected_components = sp.csgraph.connected_components(knn, directed=False)[0]

	dist = sp.csgraph.floyd_warshall(knn, directed=False)

	dist_max = np.nanmax(dist[dist != np.inf])
	dist[dist > dist_max] = 2*dist_max

	return dist, kmin


