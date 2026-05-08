import os
import json
import torch
from torch import nn
import copy
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy


class RLAlgorithm():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_action(self, state):
        '''
        return action (torch Tensor (1,action_space))
        用于继承的函数
        '''
        raise NotImplementedError

    def update_hyperparams(self, epoch):
        '''
        用于继承的函数
        '''
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        '''
        用于继承的函数
        '''
        raise NotImplementedError

    def save_params(self, time_data):
        with open(os.path.join(self.configs['current_path'], 'training_data', '{}.json'.format(time_data)), 'w') as fp:
            json.dump(self.configs, fp, indent=2)

    def load_params(self, file_name):
        ''' replay_name from flags.replay_name '''
        with open(os.path.join(self.configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
            configs = json.load(fp)
        return configs


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """存储转换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return []
        if len(self.memory) < self.capacity:
            priorities = self.priorities[:len(self.memory)]
        else:
            priorities = self.priorities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[idx] for idx in indices]
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        return transitions, indices, torch.tensor(weights, dtype=torch.float)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


def merge_dict(d1, d2):
    '''
    将d2覆盖到d1上（冲突版本）
    '''
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            print(key)
            raise KeyError
        merged[key] = d2[key]
    return merged


def merge_dict_non_conflict(d1, d2):
    '''
    将d2覆盖到d1上（非冲突版本）
    '''
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        merged[key] = d2[key]
    return merged


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target,source,configs):
    for target_param,param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-configs['tau'])+param.data*configs['tau'])