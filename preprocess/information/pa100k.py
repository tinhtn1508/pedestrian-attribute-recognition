import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """

    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'release_data/release_data/')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    assert dataset.label.shape == (100000, 26)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]
    dataset.attr_size = len(dataset.attr_name)

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 90000)  # np.array(range(80000))
    # dataset.partition.val = np.arange(90000, 100000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":

    save_dir = './dataset/'
    generate_data_description(save_dir)