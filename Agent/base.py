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
        返回动作 (torch Tensor (1,action_space))
        子类继承用的函数
        '''
        raise NotImplementedError

    def update_hyperparams(self, epoch):
        '''
        更新超参数
        子类继承用的函数
        '''
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        '''
        更新TensorBoard日志
        子类继承用的函数
        '''
        raise NotImplementedError

    def save_params(self, time_data):
        with open(os.path.join(self.configs['current_path'], 'training_data', '{}.json'.format(time_data)), 'w') as fp:
            json.dump(self.configs, fp, indent=2)

    def load_params(self, file_name):
        ''' 从flags.replay_name加载配置参数 '''
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
        """保存转换数据"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    """
    优先级经验回放(Prioritized Experience Replay, PER)
    根据TD误差的大小为每个样本分配优先级，优先采样误差大的样本
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  
        self.beta = beta  
        self.beta_increment = beta_increment  
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, *args):
        """保存转换数据"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[int(self.position)] = Transition(*args)
        self.priorities[int(self.position)] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """根据优先级采样"""
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")

        memory_length = len(self.memory)
        if memory_length < self.capacity:
            priorities = self.priorities[:memory_length]
        else:
            priorities = self.priorities

        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 采样
        indices = np.random.choice(memory_length, batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # 计算权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (memory_length * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float)

    def update_priorities(self, indices, priorities):
        """更新样本优先级"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.memory):
                self.priorities[idx] = priority
                if priority > self.max_priority:
                    self.max_priority = priority

    def __len__(self):
        return len(self.memory)


def merge_dict(d1, d2):
    '''
    将d2合并到d1上（冲突版本 - 如果有重复key会抛出异常）
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
    将d2合并到d1上（无冲突版本 - 重复key会被覆盖）
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