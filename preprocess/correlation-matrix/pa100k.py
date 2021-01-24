from loguru import logger
import numpy as np
import pickle
import os
from numpy.core.fromnumeric import shape
from scipy.io import loadmat

pa100k_data = loadmat(os.path.join('../../datasets/pa100k/annotation', 'annotation.mat'))
train_label = pa100k_data['train_label']
train_data_size = train_label.shape[0]
attributes = pa100k_data['attributes']
size_attributes = len(attributes)

nums = [0 for _ in range(size_attributes)]
for labels in train_label:
    for i in range(size_attributes):
        if labels[i] == 1:
            nums[i] += 1

adj = np.zeros((size_attributes, size_attributes), dtype=int)
for labels in train_label:
    for i in range(size_attributes):
        for j in range(size_attributes):
            if labels[i] == labels[j] and labels[i] == 1:
                adj[i][j] += 1

info = {"nums": np.array(nums), "adj": adj}
print(info)

with open('pa100k_adj.pkl', 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
