import os

from sklearn import datasets
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import metrics
import vp
from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

def load_dataset(dataset_name):
    data_dir = os.path.join('data', dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    return X, y

X, y = load_dataset('cnae9')

s = metrics.metric_dc_num_samples(X)
d = metrics.metric_dc_num_features(X)
print(s, d)

embeddings = TSNE(n_jobs=4).fit_transform(X)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()