import os
import numpy as np
import pickle
from easydict import EasyDict
from scipy.io import loadmat
import argparse
from sklearn.model_selection import train_test_split
from loguru import logger

def generate(args):
    rapv2_decription = loadmat(args.annotation_file)
    rapv2_decription = rapv2_decription['RAP_annotation']

    dataset = EasyDict()
    dataset.description = 'rapv2'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir

    dataset.image_name = [rapv2_decription['name'][0][0][i][0][0] for i in range(84928)]
    dataset.attr_name = [rapv2_decription['attribute'][0][0][i][0][0] for i in range(152)]
    raw_label = rapv2_decription['data'][0][0]
    selected_attr_idx = rapv2_decription['selected_attribute'][0][0][0] - 1

    dataset.label = raw_label[:, selected_attr_idx]

    train = []
    val = []
    test = []
    for idx in range(5):
        _train = rapv2_decription['partition_attribute'][0][0][0][idx]['train_index'][0][0][0] - 1
        _val = rapv2_decription['partition_attribute'][0][0][0][idx]['val_index'][0][0][0] - 1
        _test = rapv2_decription['partition_attribute'][0][0][0][idx]['test_index'][0][0][0] - 1
        train.append(_train)
        val.append(_val)
        test.append(_test)
    train = np.concatenate(train)
    val = np.concatenate(val)
    test = np.concatenate(test)
    data = np.concatenate([train, val, test])

    dataset.partition = EasyDict()
    train_index, test_index = train_test_split(data, shuffle=args.shuffle, train_size=args.train_rate, random_state=42)
    dataset.partition.train = train_index
    dataset.partition.test = test_index

    count_label = np.count_nonzero(dataset.label, axis=0)
    loss_weight = [i/84928 for i in count_label]
    dataset.loss_weight = np.array(loss_weight)

    with open(os.path.join(args.save_dir, 'rapv2_description.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def buildArgParse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--save_dir', default='.', type=str, required=False, help='Place to store output')
    parse.add_argument('--annotation_file', default='./RAP_annotation/RAP_annotation.mat', type=str, required=False, help='The annotation file')
    parse.add_argument('--data_dir', default='./RAP_dataset/', type=str, required=False, help='Place to put unzip data')
    parse.add_argument('--train_rate', default=0.9, type=float, required=False, help='The rate of training data')
    parse.add_argument('--shuffle', default=False, type=bool, required=False, help='train, test and validate are shuffled')
    return parse

if __name__ == "__main__":
    logger.info('Starting prepare peta dataset..........')
    parser = buildArgParse()
    args = parser.parse_args()

    generate(args)
    logger.info('Preparing process is successful !!!!!!')