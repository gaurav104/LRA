import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def JS_divergence(outputs, outputs_student):
    T = 3.0
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps = 0.0
    loss_verifier_cig = 0.5 * nn.KLDivLoss()(torch.log(P + eps), M) + 0.5 * nn.KLDivLoss()(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
    return loss_verifier_cig

def kdloss(y, teacher_scores,T=1.0):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) *(T**2)  / y.shape[0]
    return l_kl