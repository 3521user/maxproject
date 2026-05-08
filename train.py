import json
import os
from random import random
import sys
import time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import update_tensorboard
from Agent.base import merge_dict


def city_dqn_train(configs, time_data, sumoCmd):
    from Agent.super_dqn import Trainer
    if configs['model'] == 'city':
        from Env.CityEnv import CityEnv

    phase_num_matrix = torch.tensor(  # 每个交通信号灯拥有的最大相位数量
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])
    # init agent and tensorboard writer
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    agent = Trainer(configs)
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    NUM_AGENT = configs['num_agent']
    DEVICE = configs['device']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],  # i*10
                          device=DEVICE, dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=DEVICE, dtype=torch.int)
    epoch = 0
    while epoch < configs['num_epochs']:
        step = 0
        if configs['randomness'] == True:
            tmp_sumoCmd = sumoCmd+['--scale', str(1.5+random())]  # 1.5~2.5
        else:
            if configs['network'] == 'dunsan' or  'grid' in configs['network']:
                tmp_sumoCmd = sumoCmd+['--scale', str(configs['scale'])]
            else:
                tmp_sumoCmd = sumoCmd
        traci.start(tmp_sumoCmd)
        env = CityEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=DEVICE)
        # Mask矩阵：当TL_Period结束时为True
        mask_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.bool, device=DEVICE)

        # 仅在MAX Period内递增的t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=DEVICE)
        t_agent -= OFFSET

        # Action矩阵：比较相同时收集state，不存在的state用zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=DEVICE)  # 需要处理黄灯3秒
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=DEVICE)  # 当前是第几个相位
        action_update_mask = torch.eq(   # 检查action是否需要现在更新
            t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 使用0是因为索引问题

        # 达到最大值时重置为0（与offset比较）
        clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)
        t_agent[clear_matrix] = 0
        # 如果需要切换action，则增加action索引（通过tensor切片）
        action_index_matrix[action_update_mask] += 1
        action_index_matrix[clear_matrix] = 0

        # 更新mask，矩阵转换为True
        mask_matrix[clear_matrix] = True
        mask_matrix[~clear_matrix] = False

        # state initialization
        state = env.collect_state(
            action_update_mask, action_index_matrix, mask_matrix)
        total_reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()
        while step < MAX_STEPS:
            # 决定action
            actions = agent.get_action(state, mask_matrix)
            # if mask_matrix.sum()>0:
            #     print(actions.transpose(1,2))
            # 将action转换为所需形式 # 下一个需要切换的时间点矩阵
            action_matrix = env.calc_action(
                action_matrix, actions, mask_matrix)
            # 表示为累积值

            # 应用到环境
            # action应用函数，包含traci.simulationStep
            env.step(
                actions, mask_matrix, action_index_matrix, action_update_mask)

            # 整体增加1秒 # traci在env.step中处理
            step += 1
            t_agent += 1
            # 达到最大值时重置为0（与offset比较）
            clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)

            # 如果需要切换action，则增加action索引（通过tensor切片）
            for idx,_ in enumerate(TL_RL_LIST):
                action_update_mask[idx] = torch.eq(  # update只需要接收真实相位来决定
                    t_agent[idx], action_matrix[idx, action_index_matrix[idx]].view(-1))  # 使用0是因为索引问题

            action_index_matrix[action_update_mask] += 1
            # 如果超过agent的最大phase，则将该agent的action索引重置为0
            action_index_matrix[clear_matrix] = 0
            
            # 更新mask，矩阵转换为True
            t_agent[clear_matrix] = 0
            # print(t_agent,action_index_matrix,step,action_update_mask)
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False

            next_state = env.collect_state(
                action_update_mask, action_index_matrix, mask_matrix)
            # 从env中取出每个agent的state，在max_offset+period之后开始
            if step >= int(torch.max(OFFSET)+torch.max(TL_PERIOD)) and mask_matrix.sum() > 0:
                rep_state, rep_action, rep_reward, rep_next_state = env.get_state(
                    mask_matrix)
                agent.save_replay(rep_state, rep_action, rep_reward,
                                  rep_next_state, mask_matrix)  # dqn
            # update
            agent.update(mask_matrix)

            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        agent.target_update(epoch)
        agent.update_hyperparams(epoch)  # lr and epsilon upate
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        print('======== {} epoch/ return: {:.5f} arrived number:{}'.format(epoch,
                                                                           env.cum_reward.sum(), arrived_vehicles))
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        env.test_val=0
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}'.format(epoch))

    writer.close()
