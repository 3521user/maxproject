from typing import NewType
import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class Memory():
    def __init__(self, configs):
        self.configs = configs
        self.reward = torch.zeros(
            1, dtype=torch.float, device=configs['device'])
        self.state = torch.zeros(
            (1, self.configs['state_space'], 4, 1), dtype=torch.float, device=configs['device'])
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(
            (1, self.configs['action_size']), dtype=torch.int, device=configs['device'])


class CityEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.device = self.configs['device']
        self.phase_action_matrix = torch.zeros(  # 累积求和之前的动作矩阵
            (self.configs['num_agent'], self.configs['max_phase_num']), dtype=torch.int, device=self.device)  # 计算奖励时使用
        self.tl_list = traci.trafficlight.getIDList()
        self.tl_rl_list = self.configs['tl_rl_list']
        self.num_agent = len(self.tl_rl_list)
        self.side_list = ['u', 'r', 'd', 'l']
        self.interest_list = self.configs['interest_list']
        self.node_interest_pair = self.configs['node_interest_pair']

        self.reward = torch.zeros(
            (1, self.num_agent), dtype=torch.float, device=self.configs['device'])
        self.cum_reward = torch.zeros_like(self.reward)
        self.state_space = self.configs['state_space']
        self.action_size = self.configs['action_size']
        self.traffic_node_info = self.configs['traffic_node_info']
        self.nodes = self.configs['node_info']

        self.before_action_update_mask = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.device)
        self.before_action_index_matrix = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.device)
        self.tl_rl_memory = list()
        for _ in range(self.num_agent):
            self.tl_rl_memory.append(Memory(self.configs))

        # 为动作映射准备的矩阵
        self.min_phase = torch.tensor(
            self.configs['min_phase'], dtype=torch.int, device=self.device)
        self.max_phase = torch.tensor(
            self.configs['max_phase'], dtype=torch.int, device=self.device)
        self.common_phase = torch.tensor(
            self.configs['common_phase'], dtype=torch.int, device=self.device)
        self.matrix_actions = torch.tensor(
            self.configs['matrix_actions'], dtype=torch.int, device=self.device)
        # 创建相位数量列表
        self.num_phase_list = list()
        for phase in self.common_phase:
            self.num_phase_list.append(len(phase))

        self.left_lane_num_dict = dict()
        # 保存车道信息
        for interest in self.node_interest_pair:
            # 对于所有流入边
            for pair in self.node_interest_pair[interest]:
                if pair['inflow'] == None:
                    continue
                self.left_lane_num_dict[pair['inflow']] = traci.edge.getLaneNumber(
                    pair['inflow'])-1
        
        # self.test_val=list()
        # for i in self.tl_rl_list:
        #     self.test_val.append(0)

    def get_state(self, mask):
        '''
        每个周期返回该周期之前的状态、当前状态和奖励
        需要初始化 reward, next_state <- state
        '''

        state = torch.zeros(
            (1, self.state_space, 4, self.num_agent), dtype=torch.float, device=self.device)
        next_state = torch.zeros_like(state)
        action = torch.zeros(
            (1, self.num_agent, self.action_size), dtype=torch.int, device=self.device)
        reward = torch.zeros((1, self.num_agent),
                             dtype=torch.float, device=self.device)
        for index in torch.nonzero(mask):
            state[0, :, :, index] = deepcopy(self.tl_rl_memory[index].state)
            action[0, index, :] = deepcopy(self.tl_rl_memory[index].action)
            next_state[0, :, :, index] = deepcopy(
                self.tl_rl_memory[index].next_state)
            reward[0, index] = deepcopy(self.tl_rl_memory[index].reward)
            # mask index's reward clear
            self.tl_rl_memory[index].reward = 0

        return state, action, reward, next_state

    def collect_state(self, action_update_mask, action_index_matrix, mask_matrix):
        '''
        每秒检查是否有需要更新的内容
        与之前相比，索引增加并且该索引是

        基于最大压力的控制
        对于每个节点，在流入车辆数和流出车辆数+相应方向的前提下
        '''
        # 创建保存奖励的mask
        action_change_mask = torch.zeros_like(action_update_mask)
        for index in torch.nonzero(action_update_mask):
            if action_index_matrix[index] in self.traffic_node_info[self.tl_rl_list[index]]['phase_index']:
                # 当action_index_matrix上的值是需要接收next state的索引时
                action_change_mask[index] = True
                # self.test_val[index]+=1

        # Reward
        for index in torch.nonzero(action_change_mask):
            # self.test_val+=1
            outflow = 0
            inflow = 0
            interests = self.node_interest_pair[self.tl_rl_list[index]]
            for interest in interests:
                if interest['outflow']:  # 当不为None时执行
                    outflow += (traci.edge.getLastStepVehicleNumber(
                        interest['outflow']))/100.0
                if interest['inflow']:  # 当不为None时执行
                    inflow += traci.edge.getLastStepVehicleNumber(
                        interest['inflow'])/100.0
            # pressure=inflow-outflow
            # reward cumulative sum
            pressure = torch.tensor(
                abs(inflow-outflow), dtype=torch.float, device=self.device)/100.0
            self.reward[0, index] -= pressure
            self.tl_rl_memory[index].reward -= pressure

        # penalty
        for index in torch.nonzero(mask_matrix):
            if self.phase_action_matrix[index].sum() != 0:
                phase_index = torch.tensor(
                    self.traffic_node_info[self.tl_rl_list[index]]['phase_index'], device=self.device).view(1, -1).long()
                # penalty for phase duration more than maxDuration
                if torch.gt(self.phase_action_matrix[index].gather(dim=1, index=phase_index), torch.tensor(self.traffic_node_info[self.tl_rl_list[index]]['max_phase'])).sum():
                    self.reward[0, index] -= 0.4
                    self.tl_rl_memory[index].reward -= 0.4  # penalty

                # penalty for phase duration less than minDuration
                if torch.gt(torch.tensor(self.traffic_node_info[self.tl_rl_list[index]]['min_phase']), self.phase_action_matrix[index].gather(dim=1, index=phase_index)).sum():
                    self.reward[0, index] -= 0.4
                    self.tl_rl_memory[index].reward -= 0.4  # penalty

        # 为动作变化准备的状态
        next_states = torch.zeros(
            (1, self.state_space, 4, self.num_agent), dtype=torch.float, device=self.device)
        # print(action_update_mask)
        for idx in torch.nonzero(action_change_mask):
            next_state = list()
            # 对于所有rl节点
            phase_type_tensor = torch.tensor(self.configs['phase_type'][idx])
            # vehicle state
            veh_state = torch.zeros(
                (self.state_space-2-2), dtype=torch.float, device=self.device)
            for j, pair in enumerate(self.node_interest_pair[self.tl_rl_list[idx]]):
                # 对于所有流入边
                if pair['inflow'] is None:
                    veh_state[j*2] = 0.0
                    veh_state[j*2+1] = 0.0
                else:
                    left_movement = traci.lane.getLastStepVehicleNumber(
                        pair['inflow']+'_{}'.format(self.left_lane_num_dict[pair['inflow']]))/100.0  # 计算左转车辆数
                    # 直行
                    veh_state[j*2] = traci.edge.getLastStepVehicleNumber(
                        pair['inflow'])/100.0-left_movement  # 将最左侧停止的车辆视为左转车道用户
                    # 左转
                    veh_state[j*2+1] = left_movement
            # duration差异的tensor
            min_dur_tensor = torch.tensor(
                self.traffic_node_info[self.tl_rl_list[idx]]['dif_min'][int(action_index_matrix[idx]/2)], dtype=torch.float, device=self.device).view(-1)
            max_dur_tensor = torch.tensor(
                self.traffic_node_info[self.tl_rl_list[idx]]['dif_max'][int(action_index_matrix[idx]/2)], dtype=torch.float, device=self.device).view(-1)
            next_state = torch.cat((veh_state, phase_type_tensor, min_dur_tensor, max_dur_tensor), dim=0).view(
                self.state_space, 1)
            # print(next_state,idx,self.configs['phase_type'][idx])
            # print(next_state)

            self.tl_rl_memory[idx].state[:, :, (action_index_matrix[idx]/2).long()
                                         ] = self.tl_rl_memory[idx].next_state[:, :, (action_index_matrix[idx]/2).long()].detach().clone()
            self.tl_rl_memory[idx].next_state[:, :, (action_index_matrix[idx]/2).long()
                                              ] = next_state.view(1, self.state_space, 1, 1).detach().clone()

        for idx in torch.nonzero(mask_matrix):
            # 创建next state
            next_states[0, :, :,
                        idx] = self.tl_rl_memory[idx].next_state.detach().clone()
        # reward clear
        reward = self.reward.detach().clone()
        self.cum_reward += reward
        for idx, _ in enumerate(self.tl_rl_list):
            if idx not in torch.nonzero(mask_matrix).tolist():
                self.reward[0, idx] = torch.zeros_like(
                    self.reward[0, idx]).clone()
        return next_states  # 返回列表（里面包含tensor）

    def step(self, action, mask_matrix, action_index_matrix, action_update_mask):
        '''
        每秒应用动作并返回next_state
        如果yellow mask为True，则保存相应agent的奖励
        '''
        # 更新动作
        for index in torch.nonzero(mask_matrix):
            # 将动作转换为各个相位的长度
            tl_rl = self.tl_rl_list[index]
            phase_length_set = self._toPhaseLength(
                tl_rl, action[0, index])
            # 重新设置交通信号灯
            tls = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                self.tl_rl_list[index])
            for phase_idx in self.traffic_node_info[tl_rl]['phase_index']:
                tls[0].phases[phase_idx].duration = phase_length_set[phase_idx]
            traci.trafficlight.setProgramLogic(tl_rl, tls[0])
            self.tl_rl_memory[index].action = action[0, index]
            # print(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_rl_list[index])[0].phases)
            # print(phase_length_set)
        # 将动作注册到环境后观察情况，保存动作
        # 执行仿真步骤
        traci.simulationStep()
        # for index in torch.nonzero(mask_matrix):
        #     tls=traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_rl_list[index])
        #     print(tls[0].phases,"after")

        self.before_action_update_mask = action_update_mask

    def calc_action(self, action_matrix, actions, mask_matrix):
        for index in torch.nonzero(mask_matrix):
            # print(self.traffic_node_info[self.tl_rl_list[0]
            #                                              ]['phase_duration'])
            # print(actions[0])
            phase_duration_list = self.traffic_node_info[self.tl_rl_list[index]
                                                         ]['phase_duration']
            pad_mat = torch.zeros_like(action_matrix[index])
            pad_mat_size = pad_mat.size()[1]

            new_phase_duration_list = self._toPhaseLength(
                self.tl_rl_list[index], actions[0, index])
            insert_mat = torch.tensor(
                new_phase_duration_list, dtype=torch.int, device=self.device)
            mat = torch.nn.functional.pad(
                insert_mat, (0, pad_mat_size-insert_mat.size()[0]), 'constant', 0)
            action_matrix[index] = mat

            # 累积求和
            self.phase_action_matrix[index] = mat  # 累积求和前保存
            for l, _ in enumerate(phase_duration_list):
                if l >= 1:
                    action_matrix[index, l] += action_matrix[index, l-1]
            # print(action_matrix[0])
        return action_matrix.int()  # 保存累积求和

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.cum_reward.sum(),
                          self.configs['max_steps']*epoch)  # 每个epoch
        # 每个epoch清除一次值
        self.cum_reward = torch.zeros_like(self.cum_reward)

    def _toPhaseLength(self, tl_rl, action):  # 将动作转换为可解析的相位
        tl_dict = deepcopy(self.traffic_node_info[tl_rl])
        for j, idx in enumerate(tl_dict['phase_index']):
            tl_dict['phase_duration'][idx] = tl_dict['phase_duration'][idx] + \
                tl_dict['matrix_actions'][action[0, 0]][j] * \
                int((action[0, 1]+1)*1.5)
        phase_length_set = tl_dict['phase_duration']
        return phase_length_set