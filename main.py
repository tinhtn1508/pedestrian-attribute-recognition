from framework import engine
from loguru import logger
import time
import argparse
import torch
import utils
import model
from easydict import EasyDict
import framework

def loguruInitialize(workdir: str):
    logger.add(workdir + '/pedestrian_attribute_recognition_{time}.log')

def buildArgParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='.', type=str, required=False, help='Place to store log')
    parser.add_argument('--images_size', default=(256, 128), type=tuple, required=False, help='The images size')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workes')
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--epoch_step', default=[30], type=int, nargs='+', help='number of epochs to change learning rate')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch_size', default=16, type=int, help='The batch size')
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='The learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='the momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='the weight decay')
    parser.add_argument('--print_freq', default=100, type=int, help='the print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--dataset', default='./dataset/pa100k/pa100k_description.pkl', help='The path file to dataset')
    parser.add_argument('--data_workspace', default='./dataset/pa100k', help='The path file to dataset')
    return parser

def main():
    parser = buildArgParse()
    args = parser.parse_args()

    loguruInitialize(args.log_dir)
    logger.info("------------------- Main start ------------------")

    use_gpu = torch.cuda.is_available()
    train_dataset, val_dataset, num_classes, attr_name, loss_weight = utils.GetDataset(args.data_workspace, args.dataset)
    _model = model.InceptionNet(num_classes=num_classes)
    criterion = model.WeightedBinaryCrossEntropy(loss_weight)
    optimizer = torch.optim.SGD(params=_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = EasyDict()
    state.batch_size = args.batch_size
    state.lr = args.lr
    state.workers = args.workers
    state.epoch_step = args.epoch_step
    state.save_model_path = './'
    state.use_gpu = use_gpu
    state.print_freq = args.print_freq
    # Workaround
    state.attr_name = attr_name
    state.attr_num = num_classes

    engine = framework.TrainingEngine(state)
    engine.learning(_model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == "__main__":
    main()