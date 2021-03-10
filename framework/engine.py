from logging import log
from operator import index
from scipy.sparse.construct import random

from torch._C import set_flush_denormal
import model as lib
import os
from loguru import logger
import torch
import torchnet
import torchvision.transforms as transforms
from easydict import EasyDict
import utils
import shutil
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
import numpy as np
import random as rand

tqdm.monitor_interval = 0

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

        if 'image_size' not in state:
            self.__state.image_size = (256, 128)

        if 'use_pb' not in state:
            self.__state.use_pb = True

        if 'device_ids' not in state:
            self.__state.device_ids = None

        if 'evaluate' not in state:
            self.__state.evaluate = False

        self.__state.meter_loss = torchnet.meter.AverageValueMeter()
        self.__state.meter_accuracy = torchnet.meter.AverageValueMeter()
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
        self.__state.meter_accuracy.reset()

    def _onEndEpoch(self, training: bool, display=True) -> float:
        acc = self.__state.meter_accuracy.value()[0]
        if display:
            if training:
                logger.info('Epoch: [{0}]\Acc: {acc:.4f}'.format(self.__state.epoch, acc=acc))
            else:
                logger.info('Test: \Acc: {acc:.4f}'.format(acc=acc))
        return acc

    def _onStartBatch(self, training: bool, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def _onEndBatch(self, training: bool, data_loader, display=True):
        # logger.debug('Loss: {}'.format(self.__state.loss.data))
        # logger.debug('Loss\' type: {}'.format(type(self.__state.loss.data)))

        self.__state.loss_batch = self.__state.loss.data.cpu()
        self.__state.meter_loss.add(self.__state.loss_batch)
        self.__state.meter_accuracy.add(self.__state.accuracy_batch)

        if display and self.__state.print_freq != 0 and self.__state.iteration % self.__state.print_freq == 0:
            loss = self.__state.meter_loss.value()[0]
            batch_time = self.__state.batch_time.value()[0]
            data_time = self.__state.data_time.value()[0]
            accuracy = self.__state.meter_accuracy.value()[0]
            if training:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                            'Data {data_time_current:.3f} ({data_time:.3f})\t'
                            'Loss {loss_current:.4f} ({loss:.4f})\t'
                            'Accu {accuracy_current:.4f} ({accuracy:.4f})'.format(
                                self.__state.epoch, self.__state.iteration, len(data_loader),
                                batch_time_current=self.__state.batch_time_current, batch_time=batch_time,
                                data_time_current=self.__state.data_time_batch, data_time=data_time,
                                loss_current=self.__state.loss_batch, loss=loss,
                                accuracy_current=self.__state.accuracy_batch, accuracy=accuracy
                            ))
            else:
                logger.info('Test: [{1}/{2}]\t'
                            'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                            'Data {data_time_current:.3f} ({data_time:.3f})\t'
                            'Loss {loss_current:.4f} ({loss:.4f})'
                            'Accu {accuracy_current:.4f} ({accuracy:.4f})'.format(
                                self.__state.iteration, len(data_loader),
                                batch_time_current=self.__state.batch_time_current, batch_time=batch_time,
                                data_time_current=self.__state.data_time_batch, data_time=data_time,
                                loss_current=self.__state.loss_batch, loss=loss,
                                accuracy_current=self.__state.accuracy_batch, accuracy=accuracy
                            ))

    def _onForward(self, training: bool, model, criterion, optimizer=None) -> None:
        if optimizer is None:
            logger.error('Error while execute onForward')
            return

        input_var = torch.autograd.Variable(self.__state.input)
        target_var = torch.autograd.Variable(self.__state.target)
        if not training:
            input_var.volatile, target_var.volatile = True, True

        self.__state.output = model(input_var)

        if self.__state.model_name == 'inception_iccv':
            loss_list = []
            predict = self.__state.output[0]
            for output in self.__state.output:
                loss_list.append(criterion(torch.sigmoid(output), target_var))
                predict = torch.max(predict, output)
            self.__state.loss = sum(loss_list)
            self.__state.predict = predict.cpu()
            self.__state.accuracy_batch = lib.BinaryAccuracy(predict, target_var)
        elif self.__state.model_name == '':
            pass
        else:
            # in case of baseline
            self.__state.loss = criterion(self.__state.output, target_var)
            self.__state.accuracy_batch = lib.BinaryAccuracy(self.__state.output, target_var)

        if training:
            optimizer.zero_grad()
            self.__state.loss.backward()
            optimizer.step()

    def _initLearning(self, model):
        self.__state.best_score = 0

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=model.image_normalization_mean, std=model.image_normalization_std)
        if 'train_transform' not in self.__state:
            # self.__state.train_transform = transforms.Compose([
            #     MultiScaleCrop(self.__state.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            self.__state.train_transform = transforms.Compose([
                transforms.Resize(size=self.__state.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])

        if 'val_transform' not in self.__state:
            self.__state.val_transform = transforms.Compose([
                transforms.Resize(size=self.__state.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])

        logger.info('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer = None):
        self._initLearning(model)
        self.__state.train_data_convert_fn = train_dataset.toImageAttribute
        self.__state.val_data_convert_fn = val_dataset.toImageAttribute
        self.__state.model_name = model.name()

        train_dataset.transform = self.__state.train_transform
        val_dataset.transform = self.__state.val_transform

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.__state.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.__state.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=self.__state.batch_size,
                                                shuffle=False,
                                                num_workers=self.__state.workers)

        if 'resume' in self.__state:
            if os.path.isfile(self.__state.resume):
                logger.info('checkpoint loading.....')
                checkpoint = torch.load(self.__state.resume)
                self.__state.start_epoch = checkpoint['epoch']
                self.__state.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                logger.info('Loaded checkpoint \'{}\' (epoch {})'.format(
                    self.__state.evaluate, checkpoint['epoch']
                ))
            else:
                logger.error('no checkpoint found at: {}'.format(self.__state.resume))
                return

        if self.__state.use_gpu:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.__state.device_ids).cuda()
            # criterion = criterion.cuda()

        if self.__state.evaluate:
            self.test(val_loader, model, optimizer)
            return

        for epoch in range(self.__state.start_epoch, self.__state.max_epochs):
            self.__state.epoch = epoch
            lr = self._adjustLearningRate(optimizer)
            logger.info('Learning rate: {}'.format(lr))

            self.train(train_loader, model, criterion, optimizer, epoch)
            prec = self.validate(val_loader, model, criterion)
            self.test(val_loader, model, criterion)

            is_best = prec > self.__state.best_score
            self.__state.best_score = max(prec, self.__state.best_score)
            self._saveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if self.__state.use_gpu else model.state_dict(),
                'best_score': self.__state.best_score,
            }, is_best)

            logger.info('best score: {best:.3f}'.format(best=self.__state.best_score))

        return self.__state.best_core

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()

        self._onStartEpoch()

        if self.__state.use_pb:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target, _) in enumerate(data_loader):
            self.__state.iteration = i
            self.__state.data_time_batch = time.time() - end
            self.__state.data_time.add(self.__state.data_time_batch)
            self.__state.input = input
            self.__state.target = target

            self._onStartBatch(True, model, criterion, data_loader, optimizer)

            if self.__state.use_gpu:
                self.__state.target = self.__state.target.cuda()

            self._onForward(True, model, criterion, optimizer)

            self.__state.batch_time_current = time.time() - end
            self.__state.batch_time.add(self.__state.batch_time_current)
            end = time.time()

            self._onEndBatch(True, data_loader)
        self._onEndEpoch(True)

    def validate(self, data_loader, model, criterion):
        model.eval()

        self._onStartEpoch()

        if self.__state.use_pb:
            data_loader = tqdm(data_loader, desc='Validating')

        end = time.time()
        for i, (input, target, _) in enumerate(data_loader):
            self.__state.iteration = i
            self.__state.data_time_batch = time.time() - end
            self.__state.data_time.add(self.__state.data_time_batch)
            self.__state.input = input
            self.__state.target = target

            self._onStartBatch(True, model, criterion, data_loader)

            if self.__state.use_gpu:
                self.__state.target = self.__state.target.cuda()

            self._onForward(False, model, criterion, data_loader)

            self.__state.batch_time_current = time.time() - end
            self.__state.batch_time.add(self.__state.batch_time_current)
            end = time.time()

            self._onEndBatch(True, data_loader)
        score = self._onEndEpoch(True)
        return score

    def test(self, data_loader, model, criterion):
        model.eval()
        if self.__state.use_pb:
            data_loader = tqdm(data_loader, desc='Testing')

        pos_cnt = [0 for _ in range(self.__state.attr_num)]
        pos_tol = [0 for _ in range(self.__state.attr_num)]
        neg_cnt = [0 for _ in range(self.__state.attr_num)]
        neg_tol = [0 for _ in range(self.__state.attr_num)]
        accu, prec, recall, tol = 0.0, 0.0, 0.0, 0

        self._onStartEpoch()
        for _, (input, target, img_name) in enumerate(data_loader):
            self.__state.input = input
            self.__state.target = target
            if self.__state.use_gpu:
                self.__state.target = self.__state.target.cuda()
            self._onForward(False, model, criterion, data_loader)

            tol += target.size(0)
            output = torch.sigmoid(self.__state.predict).detach().numpy()
            output = np.where(output > 0.5, 1, 0)
            target = target.cpu().numpy()

            if rand.randint(0, 1000) % 100 == 0:
                index = rand.randint(0, target.shape[0]-1)
                logger.debug("Image's name: {}".format(img_name[index]))
                att = self.__state.val_data_convert_fn(output[index])
                logger.debug("Predicted attribute: {}".format(att))
                att = self.__state.val_data_convert_fn(target[index])
                logger.debug("Actual attribute: {}".format(att))

            for it in range(self.__state.attr_num):
                for jt in range(min(self.__state.batch_size, target.shape[0])):
                    if target[jt][it] == 1.:
                        pos_tol[it] += 1.
                        if output[jt][it] == 1.:
                            pos_cnt[it] += 1.

                    if target[jt][it] == 0.:
                        neg_tol[it] += 1.
                        if output[jt][it] == 0.:
                            neg_cnt[it] += 1.

            for jt in range(min(self.__state.batch_size, target.shape[0])):
                tp, fn, fp = 0., 0., 0.
                for it in range(self.__state.attr_num):
                    if output[jt][it] == 1 and target[jt][it] == 1:
                        tp += 1
                    elif output[jt][it] == 0 and target[jt][it] == 1:
                        fn += 1
                    elif output[jt][it] == 1 and target[jt][it] == 0:
                        fp += 1
                if tp + fn + fp != 0.:
                    accu +=  tp / (tp + fn + fp)
                if tp + fp != 0.:
                    prec += tp / (tp + fp)
                if tp + fn != 0:
                    recall += tp / (tp + fn)

        logger.info('=' * 100)
        logger.info('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
        mA = 0.0
        for it in range(self.__state.attr_num):
            cur_mA = ((1.0*pos_cnt[it]/(pos_tol[it]+0.0000001)) + (1.0*neg_cnt[it]/(neg_tol[it]+0.000001))) / 2.0
            mA += cur_mA
            logger.info('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(
                it, self.__state.attr_name[it], pos_cnt[it], neg_cnt[it], pos_tol[it], neg_tol[it],
                (pos_cnt[it]+neg_tol[it]-neg_cnt[it]), (neg_cnt[it]+pos_tol[it]-pos_cnt[it]), cur_mA)
                )

        mA = mA / self.__state.attr_num
        logger.info('\t' + 'mA:        '+str(mA))

        if self.__state.attr_num != 1:
            accu /= tol
            prec /= tol
            recall /= tol
            f1 = 2.0 * prec * recall / (prec + recall)
            logger.info('\t' + 'Accuracy:  '+str(accu))
            logger.info('\t' + 'Precision: '+str(prec))
            logger.info('\t' + 'Recall:    '+str(recall))
            logger.info('\t' + 'F1_Score:  '+str(f1))
        logger.info('=' * 100)

    def _saveCheckpoint(self, state, isBest, fileName='checkpoint.pth.tar'):
        if 'save_model_path' in self.__state:
            fileName = os.path.join(self.__state.save_model_path, fileName)
            if not os.path.exists(self.__state.save_model_path):
                os.makedirs(self.__state.save_model_path)

        logger.debug('Model saving....')
        torch.save(state, fileName)

        if isBest:
            fileNameBest = 'model_best_pth.tar'
            if 'save_model_path' in self.__state:
                fileNameBest = os.path.join(self.__state.save_model_path, fileNameBest)
            shutil.copyfile(fileName, fileNameBest)

    def _adjustLearningRate(self, optimizer):
        lr_list = []
        decay = 0.1 if sum(self.__state.epoch == np.array(self.__state.epoch_step)) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)
