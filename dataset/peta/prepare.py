import argparse
import os
from easydict import EasyDict
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from loguru import logger

places = [
    '3DPeS',
    'CAVIAR4REID',
    'CUHK',
    'GRID',
    'i-LID',
    'MIT',
    'PRID',
    'SARC3D',
    'TownCentre',
    'VIPeR'
]

def buildArgParse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--save_dir', default='.', type=str, required=False, help='Place to store output')
    parse.add_argument('--data_dir', default='./data', type=str, required=False, help='Place to put unzip data')
    parse.add_argument('--train_rate', default=0.9, type=float, required=False, help='The rate of training data')
    parse.add_argument('--shuffle', default=False, type=bool, required=False, help='train, test and validate are shuffled')
    return parse

def buildAttributeName(labelFiles):
    attributeName = []
    attributeMap = {}
    for labelFile in labelFiles:
        with open(labelFile, 'r') as f:
            for line in f.readlines():
                line = line[:-1]
                token = line.split(' ')
                for att in token[1:]:
                    if att not in attributeName:
                        attributeName.append(att)
                place = labelFile.split('/')[2]
                if place == 'CUHK':
                    attributeMap[place + '-' + token[0].split('.')[0]] = token[1:]
                else:
                    attributeMap[place + '-' + token[0]] = token[1:]
    assert len(attributeName) == 106

    return attributeName, attributeMap

def buildLabel(imagesName, attributeName, attributeMap):
    ret = []
    for image in imagesName:
        label = [0 for _ in range(len(attributeName))]
        token = image.split('/')
        key = token[0] + '-' + token[1][0:-4].split('_')[0]
        actualAttr = attributeMap[key]
        for index, att in enumerate(attributeName):
            if att in actualAttr:
                label[index] = 1
        ret.append(label)
    return ret


def getLabelFiles(dataDir):
    ret = []
    for place in places:
        ret.append(os.path.join(dataDir, place, 'archive/Label.txt'))
    return ret

def getImagesName(dataDir):
    ret = []
    for place in places:
        it = os.walk(os.path.join(dataDir, place))
        next(it)
        names = next(it)[2]
        for name in names:
            if name != 'Label.txt':
                ret.append(place + '/' + name)
    assert len(ret) == 19000
    return ret

def main(args):
    attrName, attrMap = buildAttributeName(getLabelFiles(args.data_dir))
    imgName = getImagesName(args.data_dir)

    labels = buildLabel(imgName, attrName, attrMap)
    assert len(labels) == 19000

    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir

    dataset.image_name = imgName
    dataset.label = np.array(labels)
    dataset.attr_name = attrName

    assert dataset.label.shape == (19000, 106)

    dataset.partition = EasyDict()
    train_index, test_index = train_test_split(np.arange(0, 19000), shuffle=args.shuffle, train_size=args.train_rate, random_state=42)
    dataset.partition.train = train_index
    dataset.partition.test = test_index

    with open(os.path.join(args.save_dir, 'peta_description.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    logger.info('Starting prepare peta dataset..........')
    parser = buildArgParse()
    args = parser.parse_args()

    main(args)
    logger.info('Preparing process is successful !!!!!!')