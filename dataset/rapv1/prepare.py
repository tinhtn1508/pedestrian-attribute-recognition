import os
import numpy as np
import pickle
from easydict import EasyDict
from scipy.io import loadmat
import argparse
from sklearn.model_selection import train_test_split
from loguru import logger

attributeName = ['Female',
                'AgeLess16',
                'Age17-30',
                'Age31-45',
                'BodyFat',
                'BodyNormal',
                'BodyThin',
                'Customer',
                'Clerk',
                'BaldHead',
                'LongHair',
                'BlackHair',
                'Hat',
                'Glasses',
                'Muffler',
                'Shirt',
                'Sweater',
                'Vest',
                'TShirt',
                'Cotton',
                'Jacket',
                'Suit-Up',
                'Tight',
                'ShortSleeve',
                'LongTrousers',
                'Skirt',
                'ShortSkirt',
                'Dress',
                'Jeans',
                'TightTrousers',
                'LeatherShoes',
                'SportShoes',
                'Boots',
                'ClothShoes',
                'CasualShoes',
                'Backpack',
                'SSBag',
                'HandBag',
                'Box',
                'PlasticBag',
                'PaperBag',
                'HandTrunk',
                'OtherAttchment',
                'Calling',
                'Talking',
                'Gathering',
                'Holding',
                'Pusing',
                'Pulling',
                'CarryingbyArm',
                'CarryingbyHand']

lossWeight = [0.311434,
            0.009980,
            0.430011,
            0.560010,
            0.144932,
            0.742479,
            0.097728,
            0.946303,
            0.048287,
            0.004328,
            0.189323,
            0.944764,
            0.016713,
            0.072959,
            0.010461,
            0.221186,
            0.123434,
            0.057785,
            0.228857,
            0.172779,
            0.315186,
            0.022147,
            0.030299,
            0.017843,
            0.560346,
            0.000553,
            0.027991,
            0.036624,
            0.268342,
            0.133317,
            0.302465,
            0.270891,
            0.124059,
            0.012432,
            0.157340,
            0.018132,
            0.064182,
            0.028111,
            0.042155,
            0.027558,
            0.012649,
            0.024504,
            0.294601,
            0.034099,
            0.032800,
            0.091812,
            0.024552,
            0.010388,
            0.017603,
            0.023446,
            0.128917]

def generate(args):
    rapv1_decription = loadmat(args.annotation_file)
    rapv1_decription = rapv1_decription['RAP_annotation']

    dataset = EasyDict()
    dataset.description = 'rapv1'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir
    dataset.attr_name =  attributeName
    dataset.loss_weight = np.array(lossWeight)

    labels = []
    imgName = []
    with open('train.txt', 'r') as f:
        for row in f.readlines():
            token = row[:-1].split(' ')
            imgName.append(token[0])
            label = [int(i) for i in token[1:]]
            labels.append(label)
    with open('test.txt', 'r') as f:
        for row in f.readlines():
            token = row[:-1].split(' ')
            imgName.append(token[0])
            label = [int(i) for i in token[1:]]
            labels.append(label)

    dataset.image_name = imgName
    dataset.label = np.array(labels)
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 33268)
    dataset.partition.test = np.arange(33268, 41585)

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