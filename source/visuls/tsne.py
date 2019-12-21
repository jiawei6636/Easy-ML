# -*- coding: utf-8 -*-
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE

def get_data():
    data = pd.read_csv("..\\data\\fncp.csv", header=None, sep=',')
    # label = [1] * 35 + [0] *86
    label="."*880+"."*880
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    labeln = [1] * 880 + [0] * 880
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(labeln[i] / 2.),
                 fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

if __name__ == '__main__':
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)