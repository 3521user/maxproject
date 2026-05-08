import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random
import os
from collections import namedtuple
from copy import deepcopy
from Agent.base import RLAlgorithm, ReplayMemory, PrioritizedReplayMemory, merge_dict, hard_update, merge_dict_non_conflict, soft_update
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
DEFAULT_CONFIG = {
    'gamma': 0.9,
    'tau': 0.001,
    'batch_size': 40,
    'experience_replay_size': 500000,
    'epsilon': 0.9,
    'epsilon_decay_rate': 0.99,
    'fc_net': [160, 160, 160, 160],
    'lr': 0.0001,
    'lr_decay_period': 50,
    'lr_decay_rate': 0.8,
    'target_update_period': 10,
    'final_epsilon': 0.0012,
    'final_lr': 0.00001,
    'alpha': 0.91,
    'cnn':[50,60],
    # Dueling DQN configs
    'dueling_dqn': True,
    # Prioritized Experience Replay configs
    'prioritized_replay': True,
    'per_alpha': 0.6,
    'per_beta': 0.4,
    'per_beta_increment': 0.001,
    # Huber Loss configs
    'use_huber_loss': True,
    'huber_delta': 1.0,
    # Reward normalization configs
    'reward_normalization': True,
    'reward_clip': (-10, 10),
    'reward_std_factor': 1.0,
    # Learning rate warmup configs
    'lr_warmup': True,
    'lr_warmup_steps': 1000,
    # Gradient clipping configs
    'grad_clip_value': 1.0,
    # N-step return configs
    'n_step': 1,
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class SuperQNetwork(nn.Module):
    def __init__(self, input_size, out_rate_size, out_time_size, configs):
        super(SuperQNetwork, self).__init__()
        self.configs = configs
        self.input_size = int(input_size)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']
        
        # 使用优先级经验回放或普通回放
        if self.configs.get('prioritized_replay', False):
            self.experience_replay = PrioritizedReplayMemory(
                self.configs['experience_replay_size'],
                alpha=self.configs.get('per_alpha', 0.6),
                beta=self.configs.get('per_beta', 0.4),
                beta_increment=self.configs.get('per_beta_increment', 0.001)
            )
        else:
            self.experience_replay = ReplayMemory(
                self.configs['experience_replay_size'])
        
        # 神经网络
        self.conv1 = nn.Conv2d(self.state_space, self.configs['cnn'][0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(self.configs['cnn'][0], self.configs['cnn'][1], kernel_size=1, bias=False)

        # 速率动作网络
        self.fc1 = nn.Linear(self.configs['cnn'][1]*4, self.configs['fc_net'][0])
        self.fc2 = nn.Linear(self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc3 = nn.Linear(self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc4 = nn.Linear(self.configs['fc_net'][2], self.configs['fc_net'][3])
        
        # Dueling DQN: 为速率动作分离价值和优势流
        if self.configs.get('dueling_dqn', False):
            self.rate_value_fc = nn.Linear(self.configs['fc_net'][3], 1)
            self.rate_advantage_fc = nn.Linear(self.configs['fc_net'][3], out_rate_size)
        else:
            self.fc5 = nn.Linear(self.configs['fc_net'][3], out_rate_size)

        # 时间动作网络
        self.fc_y1 = nn.Linear(self.configs['cnn'][1]*4+1, self.configs['fc_net'][0])  # rate+state
        self.fc_y2 = nn.Linear(self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc_y3 = nn.Linear(self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc_y4 = nn.Linear(self.configs['fc_net'][2], self.configs['fc_net'][3])
        
        # Dueling DQN: 为时间动作分离价值和优势流
        if self.configs.get('dueling_dqn', False):
            self.time_value_fc = nn.Linear(self.configs['fc_net'][3], 1)
            self.time_advantage_fc = nn.Linear(self.configs['fc_net'][3], out_time_size)
        else:
            self.fc_y5 = nn.Linear(self.configs['fc_net'][3], out_time_size)

        # 初始化权重
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.fc_y1.weight)
        nn.init.kaiming_uniform_(self.fc_y2.weight)
        nn.init.kaiming_uniform_(self.fc_y3.weight)
        nn.init.kaiming_uniform_(self.fc_y4.weight)
        
        if self.configs.get('dueling_dqn', False):
            nn.init.kaiming_uniform_(self.rate_value_fc.weight)
            nn.init.kaiming_uniform_(self.rate_advantage_fc.weight)
            nn.init.kaiming_uniform_(self.time_value_fc.weight)
            nn.init.kaiming_uniform_(self.time_advantage_fc.weight)

        if configs['mode'] == 'test':
            self.eval()

    def forward(self, input_x):
        x_cnn = f.relu(self.conv1(input_x))
        x_cnn = f.relu(self.conv2(x_cnn))
        x_cnn = x_cnn.view(-1, self.configs['cnn'][1]*4)
        
        # 速率动作流
        x_vehicle = f.relu(self.fc1(x_cnn))
        x_vehicle = f.relu(self.fc2(x_vehicle))
        x_vehicle = f.relu(self.fc3(x_vehicle))
        x_vehicle = f.relu(self.fc4(x_vehicle))
        
        if self.configs.get('dueling_dqn', False):
            # Dueling DQN: Q值分解为价值和优势 Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            rate_value = self.rate_value_fc(x_vehicle)
            rate_advantage = self.rate_advantage_fc(x_vehicle)
            rate_action_Q = rate_value + (rate_advantage - rate_advantage.mean(dim=1, keepdim=True))
        else:
            rate_action_Q = self.fc5(x_vehicle)
        
        # 时间动作流
        x_traffic = torch.cat((x_cnn, rate_action_Q.argmax(
            dim=1, keepdim=True).detach().clone()), dim=1).view(-1, self.configs['cnn'][1]*4+1)
        x_traffic = f.relu(self.fc_y1(x_traffic))
        x_traffic = f.relu(self.fc_y2(x_traffic))
        x_traffic = f.relu(self.fc_y3(x_traffic))
        x_traffic = f.relu(self.fc_y4(x_traffic))
        
        if self.configs.get('dueling_dqn', False):
            # Dueling DQN: Q值分解为价值和优势 Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            time_value = self.time_value_fc(x_traffic)
            time_advantage = self.time_advantage_fc(x_traffic)
            time_action_Q = time_value + (time_advantage - time_advantage.mean(dim=1, keepdim=True))
        else:
            time_action_Q = self.fc_y5(x_traffic)
        
        return rate_action_Q, time_action_Q


class HuberLoss(nn.Module):
    """
    Huber Loss - 平滑的L1损失，对异常值更鲁棒
    当误差小于delta时使用平方损失，否则使用线性损失
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        error = input - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        if configs['mode'] == 'train' or configs['mode'] == 'simulate':
            os.mkdir(os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model'))
            self.configs = merge_dict(configs, DEFAULT_CONFIG)
        else:  # test
            self.configs = merge_dict_non_conflict(configs, DEFAULT_CONFIG)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']

        # 动作空间
        # 速率动作空间
        self.rate_action_space = self.configs['rate_action_space']
        # 时间动作空间
        self.time_action_space = self.configs['time_action_space']
        self.action_size = self.configs['action_size']
        self.gamma = self.configs['gamma']
        self.epsilon = self.configs['epsilon']
        
        # 损失函数: Huber损失或均方误差损失
        if self.configs.get('use_huber_loss', False):
            self.criterion = HuberLoss(delta=self.configs.get('huber_delta', 1.0))
        else:
            self.criterion = nn.MSELoss()
            
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.batch_size = self.configs['batch_size']
        self.device = self.configs['device']
        self.running_loss = 0
        
        # 学习率预热变量
        self.warmup_steps = self.configs.get('lr_warmup_steps', 1000)
        self.warmup_current_step = 0
        self.initial_lr = self.configs['lr']
        
        # 奖励归一化变量
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        self.reward_epsilon = 1e-8
        
        # 梯度裁剪值
        self.grad_clip_value = self.configs.get('grad_clip_value', 1.0)

        # 神经网络组合
        self.rate_key_list = list()
        for i, key in enumerate(self.configs['traffic_node_info'].keys()):
            if configs['mode'] == 'train':
                rate_key = self.configs['traffic_node_info'][key]['num_phase']
            elif configs['mode'] == 'test':
                rate_key = str(
                    self.configs['traffic_node_info'][key]['num_phase'])
            self.rate_key_list.append(rate_key)

        self.mainSuperQNetwork = SuperQNetwork(
            self.state_space, self.rate_action_space[rate_key], self.time_action_space[0], self.configs)
        self.targetSuperQNetwork = SuperQNetwork(
            self.state_space, self.rate_action_space[rate_key], self.time_action_space[0], self.configs)
        
        # 优化器: AdamW或Adadelta
        if self.configs.get('use_adamw', False):
            self.optimizer = optim.AdamW(
                self.mainSuperQNetwork.parameters(), 
                lr=self.configs['lr'],
                weight_decay=self.configs.get('weight_decay', 1e-4)
            )
        else:
            self.optimizer = optim.Adadelta(
                self.mainSuperQNetwork.parameters(), lr=self.configs['lr'])
        
        hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, 
            step_size=self.configs['lr_decay_period'], 
            gamma=self.configs['lr_decay_rate']
        )
    
    def normalize_reward(self, reward):
        """
        奖励归一化 - 使用在线均值和方差进行归一化
        """
        if not self.configs.get('reward_normalization', False):
            return reward
        
        # 更新运行均值和方差 (Welford算法)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += delta * delta2
        
        # 计算标准差
        std = np.sqrt(self.reward_var / self.reward_count) if self.reward_count > 1 else 1.0
        
        # 归一化奖励
        normalized_reward = (reward - self.reward_mean) / (std + self.reward_epsilon)
        
        # 如果指定则裁剪奖励
        if 'reward_clip' in self.configs:
            clip_min, clip_max = self.configs['reward_clip']
            normalized_reward = torch.clamp(normalized_reward, clip_min, clip_max)
        
        return normalized_reward
    
    def apply_lr_warmup(self):
        """
        学习率预热 - 在训练初期逐步增加学习率
        """
        if not self.configs.get('lr_warmup', False):
            return
        
        if self.warmup_current_step < self.warmup_steps:
            warmup_factor = float(self.warmup_current_step) / float(max(1, self.warmup_steps))
            current_lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            self.warmup_current_step += 1

    def get_action(self, state, mask):
        # 使用epsilon贪婪策略选择动作
        actions = torch.zeros((1, self.num_agent, self.action_size),
                              dtype=torch.int, device=self.device)
        with torch.no_grad():
            rate_actions = torch.zeros(
                (1, self.num_agent, 1), dtype=torch.int, device=self.device)
            time_actions = torch.zeros(
                (1, self.num_agent, 1), dtype=torch.int, device=self.device)
            for index in torch.nonzero(mask):
                if self.configs['mode'] == 'train':
                    if random.random() > self.epsilon:  # epsilon贪婪
                        # masks = torch.cat((mask, mask), dim=0)
                        rate_action, time_action = self.mainSuperQNetwork(
                            state[0, :, :, index].view(-1, self.state_space, 4, 1))
                        rate_actions[0, index] = rate_action.max(1)[1].int()
                        time_actions[0, index] = time_action.max(1)[1].int()
                        # 如果agent增加，则使用view(agents,action_size)
                    else:
                        rate_actions[0, index] = torch.tensor(random.randint(
                            0, self.rate_action_space[self.rate_key_list[index]]-1), dtype=torch.int, device=self.device)
                        time_actions[0, index] = torch.tensor(random.randint(
                            0, self.configs['time_action_space'][index]-1), dtype=torch.int, device=self.device)
                else:  # test
                    rate_action, time_action = self.mainSuperQNetwork(
                        state[0, :,:, index].view(-1, self.state_space,4, 1))
                    rate_actions[0, index] = rate_action.max(1)[1].int()
                    time_actions[0, index] = time_action.max(1)[1].int()

            actions = torch.cat((rate_actions, time_actions), dim=2)
        return actions

    def target_update(self,epoch):
        # 硬更新
        if epoch%self.configs['target_update_period']==0 and self.configs['update_type']=='hard':
            hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)

        # # 软更新
        # 全部更新
        if self.configs['update_type']=='soft':
            soft_update(self.targetSuperQNetwork,
                        self.mainSuperQNetwork, self.configs)

    def save_replay(self, state, action, reward, next_state, mask):
        for index in torch.nonzero(mask):
            # print(state[0,:, index])#,action[0,index],reward[0,index].sum(),next_state[0,:, index].sum())
            self.mainSuperQNetwork.experience_replay.push(
                state[0, :, :, index].view(-1, self.state_space, 4, 1), action[0, index], reward[0, index], next_state[0, :, :, index].view(-1, self.state_space, 4, 1))
            # print("state {}".format(state[0, :, :, index].view(-1, self.state_space, 4, 1)))
            # print("action {}".format(action[0, index]))
            # print("reward {}".format(reward[0, index]))
            # print("next_state {}".format(next_state[0, :, :, index].view(-1, self.state_space, 4, 1)))
            # if torch.eq(state[0, :, :, index],next_state[0, :, :, index]).sum()>0:
            #     print(torch.eq(state[0, :, :, index],next_state[0, :, :, index]).sum())
            #     print("FAKE")

    def update(self, mask):
        if len(self.mainSuperQNetwork.experience_replay) > self.configs['batch_size'] and mask.sum() > 0:
            # 应用学习率预热
            self.apply_lr_warmup()
            
            # 从重放内存中采样 (优先级或普通)
            if self.configs.get('prioritized_replay', False):
                transitions, indices, weights = self.mainSuperQNetwork.experience_replay.sample(
                    self.configs['batch_size'])
                weights = weights.to(self.device)
            else:
                transitions = self.mainSuperQNetwork.experience_replay.sample(
                    self.configs['batch_size'])
                indices = None
                weights = None

            batch = Transition(*zip(*transitions))

            # 计算非终止状态掩码和下一状态
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None], dim=0)

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            # 应用奖励归一化
            if self.configs.get('reward_normalization', False):
                reward_batch = self.normalize_reward(reward_batch)

            # 计算 Q(s_t, a)
            rate_state_action_values, time_state_action_values = self.mainSuperQNetwork(
                state_batch)
            rate_state_action_values = rate_state_action_values.gather(
                1, action_batch[:, 0].view(-1, 1).long())
            time_state_action_values = time_state_action_values.gather(
                1, action_batch[:, 1].view(-1, 1).long())
            
            # 计算 V(s_{t+1})
            rate_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            time_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            
            if non_final_mask.sum() > 0:
                rate_Q, time_Q = self.targetSuperQNetwork(non_final_next_states)

                # DDQN: 使用主网络选择动作，目标网络评估
                behavior_actions = self.mainSuperQNetwork(non_final_next_states)
                behavior_rate_action = torch.argmax(behavior_actions[0], dim=1, keepdim=True)
                behavior_time_action = torch.argmax(behavior_actions[1], dim=1, keepdim=True)
                rate_next_state_values[non_final_mask] = rate_Q.detach().gather(
                    dim=1, index=behavior_rate_action).view(-1)
                time_next_state_values[non_final_mask] = time_Q.detach().gather(
                    dim=1, index=behavior_time_action).view(-1)

            # 计算期望Q值
            rate_expected_state_action_values = (
                rate_next_state_values * (self.configs['gamma'] ** self.configs.get('n_step', 1))) + reward_batch
            time_expected_state_action_values = (
                time_next_state_values * (self.configs['gamma'] ** self.configs.get('n_step', 1))) + reward_batch

            # 计算损失
            rate_td_error = rate_state_action_values - rate_expected_state_action_values.unsqueeze(1)
            time_td_error = time_state_action_values - time_expected_state_action_values.unsqueeze(1)
            
            if self.configs.get('prioritized_replay', False) and weights is not None:
                # 优先级经验回放的加权损失
                rate_loss = (self.criterion(rate_state_action_values,
                                           rate_expected_state_action_values.unsqueeze(1)) * weights.unsqueeze(1)).mean()
                time_loss = (self.criterion(time_state_action_values,
                                           time_expected_state_action_values.unsqueeze(1)) * weights.unsqueeze(1)).mean()
            else:
                rate_loss = self.criterion(rate_state_action_values,
                                           rate_expected_state_action_values.unsqueeze(1))
                time_loss = self.criterion(time_state_action_values,
                                           time_expected_state_action_values.unsqueeze(1))
            
            self.running_loss += rate_loss.item() / self.configs['batch_size']
            self.running_loss += time_loss.item() / self.configs['batch_size']

            # 优化模型
            self.optimizer.zero_grad()
            total_loss = rate_loss + time_loss
            total_loss.backward()
            
            # 带None检查的增强梯度裁剪
            self.clip_gradients()
            
            self.optimizer.step()

            # 更新优先级经验回放的优先级
            if self.configs.get('prioritized_replay', False) and indices is not None:
                td_errors = torch.cat([rate_td_error.detach().abs(), time_td_error.detach().abs()], dim=1)
                td_errors = td_errors.max(dim=1)[0].cpu().numpy()
                self.mainSuperQNetwork.experience_replay.update_priorities(indices, td_errors)
    
    def clip_gradients(self):
        """
        带None检查的增强梯度裁剪，确保安全
        """
        grad_clip_value = self.configs.get('grad_clip_value', 1.0)
        
        for param in self.mainSuperQNetwork.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip_value, grad_clip_value)

    def update_hyperparams(self, epoch):
        # 衰减率 (epsilon贪婪)
        if self.epsilon > self.configs['final_epsilon']:
            self.epsilon *= self.epsilon_decay_rate

        # 衰减学习率
        # if self.lr > self.configs['final_lr']:
        #     self.lr = self.lr_decay_rate*self.lr

        self.lr_scheduler.step()

    def save_weights(self, name):

        torch.save(self.mainSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super.h5'))
        torch.save(self.targetSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super_target.h5'))

    def load_weights(self, name):
        # 加载预训练模型权重
        # 模型保存在训练时的时间戳目录下，而不是测试时间戳目录
        model_path = os.path.join(
            self.configs['current_path'], 'training_data', name, 'model', name+'_{}Super.h5'.format(self.configs['replay_epoch']))
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 列出目录中所有可用的模型文件
            model_dir = os.path.join(self.configs['current_path'], 'training_data', name, 'model')
            if os.path.exists(model_dir):
                available_models = [f for f in os.listdir(model_dir) if f.endswith('Super.h5') and not f.endswith('_target.h5')]
                raise FileNotFoundError(
                    f"\n[ERROR] 找不到指定的模型文件: {model_path}\n"
                    f"可用的模型文件:\n{chr(10).join(['  - ' + m for m in available_models])}\n"
                    f"请检查 --replay_name 和 --replay_epoch 参数是否正确"
                )
            else:
                raise FileNotFoundError(f"\n[ERROR] 模型目录不存在: {model_dir}")
        
        print(f"[INFO] 正在加载模型: {model_path}")
        state_dict = torch.load(model_path)
        new_state_dict = self.mainSuperQNetwork.state_dict()
        
        # 检测模型结构类型
        model_type = "unknown"
        if any(key.startswith('rate_value_fc') or key.startswith('time_value_fc') for key in state_dict.keys()):
            model_type = "dueling_dqn"
        elif any(key.startswith('fc5') or key.startswith('fc_y5') for key in state_dict.keys()):
            model_type = "old_fc5"
        elif any(key.startswith('fc_v') or key.startswith('fc_a_rate') for key in state_dict.keys()):
            model_type = "old_fc_v"
        
        print(f"[INFO] Detected model type: {model_type}")
        
        if model_type == "dueling_dqn":
            # 新模型结构，直接加载
            self.mainSuperQNetwork.load_state_dict(state_dict)
        else:
            # 旧模型结构，进行权重映射
            # 首先复制所有匹配的权重
            for key in state_dict.keys():
                if key in new_state_dict:
                    new_state_dict[key] = state_dict[key]
            
            # 对于Dueling DQN新增的层，使用适当的初始化
            # 从旧模型中相似的层复制权重
            if model_type == "old_fc5":
                if 'fc4.weight' in state_dict:
                    self._init_new_layers_from_old(new_state_dict, state_dict, 'fc4')
            elif model_type == "old_fc_v":
                if 'fc_v.weight' in state_dict:
                    self._init_new_layers_from_old(new_state_dict, state_dict, 'fc_v')
            
            self.mainSuperQNetwork.load_state_dict(new_state_dict, strict=False)
            print(f"[WARNING] Loaded old model structure ({model_type}), some layers were randomly initialized")
        
        self.mainSuperQNetwork.eval()
    
    def _init_new_layers_from_old(self, new_state_dict, old_state_dict, source_layer_prefix):
        """从旧模型层初始化新模型的Dueling层"""
        # rate分支的value和advantage层
        if f'{source_layer_prefix}.weight' in old_state_dict:
            source_weight = old_state_dict[f'{source_layer_prefix}.weight']
            
            # 只在形状兼容时复制权重，否则保持随机初始化
            if 'rate_value_fc.weight' in new_state_dict:
                target_size = new_state_dict['rate_value_fc.weight'].size()
                if source_weight.size(1) == target_size[1]:
                    # 从源层复制到value层（只取需要的行数）
                    new_state_dict['rate_value_fc.weight'] = source_weight[:target_size[0]].clone()
            
            if 'rate_advantage_fc.weight' in new_state_dict:
                target_size = new_state_dict['rate_advantage_fc.weight'].size()
                if source_weight.size(1) == target_size[1]:
                    # 只在形状兼容时复制，否则保持随机初始化
                    if source_weight.size(0) >= target_size[0]:
                        new_state_dict['rate_advantage_fc.weight'] = source_weight[:target_size[0]].clone()
            
            if 'time_value_fc.weight' in new_state_dict:
                target_size = new_state_dict['time_value_fc.weight'].size()
                if source_weight.size(1) == target_size[1]:
                    new_state_dict['time_value_fc.weight'] = source_weight[:target_size[0]].clone()
            
            if 'time_advantage_fc.weight' in new_state_dict:
                target_size = new_state_dict['time_advantage_fc.weight'].size()
                if source_weight.size(1) == target_size[1]:
                    if source_weight.size(0) >= target_size[0]:
                        new_state_dict['time_advantage_fc.weight'] = source_weight[:target_size[0]].clone()
        
        # 复制偏置
        if f'{source_layer_prefix}.bias' in old_state_dict:
            source_bias = old_state_dict[f'{source_layer_prefix}.bias']
            
            if 'rate_value_fc.bias' in new_state_dict:
                target_size = new_state_dict['rate_value_fc.bias'].size(0)
                if source_bias.size(0) >= target_size:
                    new_state_dict['rate_value_fc.bias'] = source_bias[:target_size].clone()
            
            if 'rate_advantage_fc.bias' in new_state_dict:
                target_size = new_state_dict['rate_advantage_fc.bias'].size(0)
                if source_bias.size(0) >= target_size:
                    new_state_dict['rate_advantage_fc.bias'] = source_bias[:target_size].clone()
            
            if 'time_value_fc.bias' in new_state_dict:
                target_size = new_state_dict['time_value_fc.bias'].size(0)
                if source_bias.size(0) >= target_size:
                    new_state_dict['time_value_fc.bias'] = source_bias[:target_size].clone()
            
            if 'time_advantage_fc.bias' in new_state_dict:
                target_size = new_state_dict['time_advantage_fc.bias'].size(0)
                if source_bias.size(0) >= target_size:
                    new_state_dict['time_advantage_fc.bias'] = source_bias[:target_size].clone()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 每个epoch
        writer.add_scalar('hyperparameter/lr', self.optimizer.param_groups[0]['lr'],
                          self.configs['max_steps']*epoch)
        writer.add_scalar('hyperparameter/epsilon',
                          self.epsilon, self.configs['max_steps']*epoch)
        # writer.add_histogram('action/time',self.time_action_dist_save, self.configs['max_steps']*epoch)
        # writer.add_histogram('action/rate',self.rate_action_dist_save, self.configs['max_steps']*epoch)

        # clear
        self.running_loss = 0