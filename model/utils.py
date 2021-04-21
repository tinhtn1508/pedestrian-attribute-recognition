import torch
import torch.nn as nn
import numpy as np
from loguru import logger

EPS = 1e-12

class WeightedBinaryCrossEntropy(object):
    def __init__(self, weight: np.array):
        self.__weight = torch.Tensor(weight).cuda()

    def _forward(self, output, target):
        if self.__weight is None:
            logger.error('the weight can not None')
            raise Exception('Error in WeightedBinaryCrossEntropy')

        cur_weights = torch.exp(target + (1 - target * 2) * self.__weight)
        loss = cur_weights *  (target * torch.log(output + EPS)) + ((1 - target) * torch.log(1 - output + EPS))
        return torch.neg(torch.mean(loss))

    def __call__(self, output, target):
        return self._forward(output, target)

def BinaryAccuracy(output, target) -> float:
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output.cpu()).detach().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num