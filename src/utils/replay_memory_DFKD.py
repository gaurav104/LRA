"""
Replay Memory class
Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
date: 1st of June 2018
"""
import random
from collections import namedtuple

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, config):
        self.config = config

        self.capacity = self.config.memory_capacity
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, batch):
        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = batch
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer

    def sample_batch(self):
        batch = random.choice(self.memory)
        return batch


class ReplayMemoryContinuous(object):
    def __init__(self, config):
        self.config = config

        self.capacity = self.config.memory_capacity * self.config.batch_size // 8

        try:
            num_channels = self.config.num_channels 
        except:
            num_channels = 3

        self.memory = torch.tensor([])
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, batch):
        if self.length() < self.capacity:
            self.memory  = torch.cat((self.memory, torch.zeros(batch.shape, device='cpu')))
        self.memory[self.position*batch_size: (self.position+1)*batch_size] = batch

        self.position = (self.position + 1) % (self.config.memory_capacity)  # for the cyclic buffer

    def sample_batch(self):
        batch = random.choice(self.memory)
        return batch