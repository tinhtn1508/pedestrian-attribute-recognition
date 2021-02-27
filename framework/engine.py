from logging import log
from loguru import logger
import torch
import torchnet
from easydict import EasyDict

class TrainingEngine():
    def __init__(self, state: EasyDict):
        self.__state = state

        if 'use_gpu' not in state:
            self.__state.use_gpu = torch.cuda.is_available()

        if 'batch_size' not in state:
            self.__state.batch_size = 32

        if 'workers' not in state:
            self.__state.workers = 8

        if 'start_epoch' not in state:
            self.__state.start_epoch = 0

        if 'max_epochs' not in state:
            self.__state.max_epochs = 60

        if 'print_freq' not in state:
            self.__state.print_freq = 0

        self.__state.meter_loss = torchnet.meter.AverageValueMeter()
        self.__state.batch_time = torchnet.meter.AverageValueMeter()
        self.__state.data_time = torchnet.meter.AverageValueMeter()

        self.__printState()

    def __printState(self):
        logger.info('----------------------------------------------')
        logger.info('Current state: ')
        for k, v in self.__state.items():
            logger.info('*** {}:\t{}'.format(k, v))
        logger.info('----------------------------------------------')

    def _onStartEpoch(self):
        self.__state.meter_loss.reset()
        self.__state.batch_time.reset()
        self.__state.data_time.reset()

    def _onEndEpoch(self, training: bool, display=True) -> float:
        loss = self.__state.meter_loss.value()[0]
        if display:
            if training:
                logger.info('Epoch: [{0}]\tLoss: {loss:.4f}'.format(self.__state.epoch, loss=loss))
            else:
                logger.info('Test: \nLoss: {loss:.4f}'.format(loss=loss))
        return

    def _onStartBatch(self, training: bool, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def _onEndBatch(self, training: bool, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def _onForward(self, training: bool, model, criterion, optimizer) -> None:
        if optimizer is None:
            logger.error('Error while execute onForward')
            return

        input_var = torch.autograd.Variable(self.__state.input)
        target_var = torch.autograd.Variable(self.__state.target)
        if not training:
            input_var.volatile. target_var.volatile = True, True

        self.__state.output = model(input_var)
        self.__state.loss = criterion(self.__state.output, target_var)

        if training:
            optimizer.zero_grad()
            self.__state.loss.backward()
            optimizer.step()

    def initLearning(self, model, criterion):
        pass

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer = None):
        pass

    def train(self, data_loader, model, criterion, optimizer, epoch):
        pass

    def validate(self, data_loader, model, criterion):
        pass

    def saveCheckpoint(self, state, isBest, fileName='checkpoint.pth.tar'):
        pass

    def adjustLearningRate(self, optimizer):
        pass