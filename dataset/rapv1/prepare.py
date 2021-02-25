import os
import numpy as np
import pickle
from easydict import EasyDict
from scipy.io import loadmat
import argparse
from sklearn.model_selection import train_test_split
from loguru import logger

def generate(args):
    rapv1_decription = loadmat(args.annotation_file)
    rapv1_decription = rapv1_decription['RAP_annotation']

    dataset = EasyDict()
    dataset.description = 'rapv1'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir

    dataset.image_name = [rapv1_decription[0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [rapv1_decription[0][0][3][i][0][0] for i in range(92)]
    raw_label = rapv1_decription[0][0][1]
    dataset.label = raw_label[:, np.array(range(51))]
    dataset.attr_name = [raw_attr_name[i] for i in range(51)]

    train = []
    test = []
    for idx in range(5):
        _train = rapv1_decription[0][0][0][idx][0][0][0][0][0, :] - 1
        _test = rapv1_decription[0][0][0][idx][0][0][0][1][0, :] - 1
        train.append(_train)
        test.append(_test)
    train = np.concatenate(train)
    test = np.concatenate(test)
    data = np.concatenate([train, test])
    print(data.shape)

    dataset.partition = EasyDict()
    train_index, test_index = train_test_split(data, shuffle=args.shuffle, train_size=args.train_rate, random_state=42)
    dataset.partition.train = train_index
    dataset.partition.test = test_index

    with open(os.path.join(args.save_dir, 'rapv1_description.pkl'), 'wb+') as f:
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