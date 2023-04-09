import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import num_classes, num_channels


class LearningRate(nn.Module):

	def __init__(self, scalar_tensor):
		super(LearningRate, self).__init__()

		self.lr = nn.Parameter(torch.tensor(scalar_tensor))


	def forward(self):

		lr = torch.abs(self.lr) 
		out = torch.clamp(lr, min=1e-8, max=1.0)

		return out


