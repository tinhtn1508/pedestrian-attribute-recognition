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

attributeName = ['Age16-30',
                'Age31-45',
                'Age46-60',
                'AgeAbove61',
                'Backpack',
                'CarryingOther',
                'Casual lower',
                'Casual upper',
                'Formal lower',
                'Formal upper',
                'Hat',
                'Jacket',
                'Jeans',
                'Leather Shoes',
                'Logo',
                'Long hair',
                'Male',
                'Messenger Bag',
                'Muffler',
                'No accessory',
                'No carrying',
                'Plaid',
                'PlasticBags',
                'Sandals',
                'Shoes',
                'Shorts',
                'Short Sleeve',
                'Skirt',
                'Sneaker',
                'Stripes',
                'Sunglasses',
                'Trousers',
                'Tshirt',
                'UpperOther',
                'V-Neck']

loss_weight = [ 0.5016,
                0.3275,
                0.1023,
                0.0597,
                0.1986,
                0.2011,
                0.8643,
                0.8559,
                0.1342,
                0.1297,
                0.1014,
                0.0685,
                0.314,
                0.2932,
                0.04,
                0.2346,
                0.5473,
                0.2974,
                0.0849,
                0.7523,
                0.2717,
                0.0282,
                0.0749,
                0.0191,
                0.3633,
                0.0359,
                0.1425,
                0.0454,
                0.2201,
                0.0178,
                0.0285,
                0.5125,
                0.0838,
                0.4605,
                0.0124]

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
        key = token[0] + '-' + token[2][0:-4].split('_')[0]
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
                ret.append(place + '/archive/' + name)
    assert len(ret) == 19000
    return ret

def main(args):
    dataset = EasyDict()
    dataset.description = 'peta'
    dataset.reorder = 'group_order'
    dataset.root = args.data_dir
    dataset.attr_name =  attributeName
    dataset.loss_weight = np.array(loss_weight)
    imgName = []
    labels = []
    with open('PETA_train_list.txt', 'r') as f:
        for row in f.readlines():
            token = row.split(' ')[:-1]
            label = [int(i) for i in token[1:]]
            imgName.append(token[0])
            labels.append(label)

    with open('PETA_test_list.txt', 'r') as f:
        for row in f.readlines():
            token = row.split(' ')[:-1]
            label = [int(i) for i in token[1:]]
            imgName.append(token[0])
            labels.append(label)

    dataset.image_name = imgName
    dataset.label = np.array(labels)
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 11400)
    dataset.partition.test = np.arange(11400, 19000)

    with open(os.path.join(args.save_dir, 'peta_description.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    logger.info('Starting prepare peta dataset..........')
    parser = buildArgParse()
    args = parser.parse_args()

    main(args)
    logger.info('Preparing process is successful !!!!!!')