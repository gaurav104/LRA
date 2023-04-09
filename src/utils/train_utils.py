import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import math

"""
Learning rate adjustment used for CondenseNet model training
"""
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


"""
Hook for batch normalization layer
"""
class hook_for_BNLoss():

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        T_mean = module.running_mean.data
        T_var = module.running_var.data
        self.G_kd_loss = self.Gaussian_kd(mean, var, T_mean, T_var)

    def Gaussian_kd(self, mean, var, T_mean, T_var):

        num = (mean-T_mean)**2 + var
        denom = 2*T_var
        std = torch.sqrt(var)
        T_std = torch.sqrt(T_var)

        return num/denom - torch.log(std/T_std) - 0.5

    def close(self):
        self.hook.remove()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']