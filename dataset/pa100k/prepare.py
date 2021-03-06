import os
import numpy as np
import pickle
from easydict import EasyDict
from scipy.io import loadmat
import argparse
from sklearn.model_selection import train_test_split
from loguru import logger
import math

def generate(args):
    pa100k_decription = loadmat(args.annotation_file)

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir

    train_image_name = [pa100k_decription['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_decription['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_decription['test_images_name'][i][0][0] for i in range(10000)]

    dataset.image_name = train_image_name + val_image_name + test_image_name
    dataset.label = np.concatenate((pa100k_decription['train_label'], pa100k_decription['val_label'], pa100k_decription['test_label']), axis=0)
    dataset.attr_name = [pa100k_decription['attributes'][i][0][0] for i in range(26)]
    assert dataset.label.shape == (100000, 26)

    dataset.partition = EasyDict()
    train_index, test_index = train_test_split(np.arange(0, 100000), shuffle=args.shuffle, train_size=args.train_rate, random_state=42)
    dataset.partition.train = train_index
    dataset.partition.test = test_index

    # the loss weight
    count_label = np.count_nonzero(dataset.label, axis=0)
    loss_weight = [i/100000 for i in count_label] 
    dataset.loss_weight = np.array(loss_weight)

    with open(os.path.join(args.save_dir, 'pa100k_description.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def buildArgParse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--save_dir', default='.', type=str, required=False, help='Place to store output')
    parse.add_argument('--annotation_file', default='./annotation.mat', type=str, required=False, help='The annotation file')
    parse.add_argument('--data_dir', default='./release_data/release_data/', type=str, required=False, help='Place to put unzip data')
    parse.add_argument('--train_rate', default=0.9, type=float, required=False, help='The rate of training data')
    parse.add_argument('--shuffle', default=False, type=bool, required=False, help='train, test and validate are shuffled')
    return parse

if __name__ == "__main__":
    logger.info('Starting prepare pa100k dataset.............')
    parser = buildArgParse()
    args = parser.parse_args()

    generate(args)
    logger.info('Preparing process is successful !!!!!!')