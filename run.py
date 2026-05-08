import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import traci
import random
import numpy as np
import traci.constants as tc
from sumolib import checkBinary
from configs import EXP_CONFIGS
from Agent.base import merge_dict, merge_dict_non_conflict


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="选择运行模式",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train或test或simulate或fixtime，"train_old"是旧版本训练模式，"fixtime"是固定时间控制模式用于对比')
    parser.add_argument(
        '--network', type=str, default='5x5grid',
        help='选择网络环境或从map文件加载')
    # optional input parameters
    parser.add_argument(
        '--disp', type=bool, default=False,
        help='训练时显示过程')
    parser.add_argument(
        '--algorithm', type=str, default='super_dqn',
        help='选择算法super_dqn')
    parser.add_argument(
        '--model', type=str, default='base',
        help='选择模型"city"')
    parser.add_argument(
        '--gpu', type=bool, default=False,
        help='选择GPU或CPU')
    parser.add_argument(
        '--replay_name', type=str, default=None,
        help='仅在测试模式下激活，输入要加载权重的文件名')
    parser.add_argument(
        '--replay_epoch', type=str, default=None,
        help='仅在测试模式下激活，输入要加载权重的文件名')
    parser.add_argument(
        '--randomness', type=bool, default=False,
        help='仅在测试模式下激活，输入要加载权重的文件名')
    parser.add_argument(
        '--update_type', type=str, default='soft', help='hard或soft')
    return parser.parse_known_args(args)[0]


def train(flags, time_data, configs, sumoConfig):

    # check gui option
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    # configs setting
    configs['num_agent'] = len(configs['tl_rl_list'])
    configs['algorithm'] = flags.algorithm.lower()
    configs['randomness'] = flags.randomness
    print("training algorithm: ", configs['algorithm'])
    configs['action_size'] = 2
    # state space由map.py决定
    if flags.network.lower() == 'grid':
        configs['state_space'] = 10

    configs['model'] = 'city'
    from train import city_dqn_train
    from configs import SUPER_DQN_TRAFFIC_CONFIGS
    configs = merge_dict_non_conflict(configs, SUPER_DQN_TRAFFIC_CONFIGS)
    city_dqn_train(configs, time_data, sumoCmd)


def test(flags, configs, sumoConfig):
    from utils import save_params, load_params, update_tensorboard
    from test import city_dqn_test
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--scale", configs['scale']]

    if flags.algorithm.lower() == 'super_dqn':
        city_dqn_test(flags, sumoCmd, configs)


def simulate(flags, configs, sumoConfig):
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--scale", configs['scale']]
    MAX_STEPS = configs['max_steps']
    traci.start(sumoCmd)
    a = time.time()
    traci.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_NUMBER])
    # traci.edge.subscribe('n_2_2_to_n_2_1', [
    #                      tc.LAST_STEP_VEHICLE_HALTING_NUMBER], 0, 2000)
    avg_velocity = 0
    step = 0
    # agent setting
    arrived_vehicles = 0
    avg_velocity = 0
    part_velocity = list()
    # travel time
    i = 0
    total_velocity = list()
    # travel time
    travel_time = list()
    waiting_time = list()
    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1
        # check performance
        for _, interests in enumerate(configs['interest_list']):
            # 删除重复
            dup_list = list()
            for interest in interests:
                inflow = interest['inflow']
                outflow = interest['outflow']
                # 信号组流向
                if inflow != None and inflow not in dup_list:
                    # 车辆的等待时间，只在有车辆时
                    if traci.edge.getLastStepVehicleNumber(inflow) != 0:
                        # 以waiting time方式append后计算avg
                        # /float(
                        waiting_time.append(traci.edge.getWaitingTime(inflow))
                        # traci.edge.getLastStepVehicleNumber(inflow)))
                        # 车辆的平均速度
                        # part_velocity.append(
                        #     traci.edge.getLastStepMeanSpeed(inflow))
                        tmp_travel = traci.edge.getTraveltime(inflow)
                        if tmp_travel <= 500 and tmp_travel != -1:  # 过滤异常值
                            travel_time.append(tmp_travel)
                        # print(travel_time)
                    dup_list.append(inflow)

                if outflow != None and outflow not in dup_list:
                    if traci.edge.getLastStepVehicleNumber(outflow) != 0:
                        # part_velocity.append(
                        #     traci.edge.getLastStepMeanSpeed(interest['outflow']))
                        tmp_travel = traci.edge.getTraveltime(outflow)
                        if tmp_travel <= 500 and tmp_travel != -1:  # 过滤异常值
                            travel_time.append(tmp_travel)
                    dup_list.append(interest['outflow'])

        # edge_list=traci.edge.getIDList()
        # for edgeid in edge_list:
        #     if traci.edge.getLastStepVehicleNumber(edgeid) !=None:
        #         total_velocity.append(traci.edge.getLastStepMeanSpeed(edgeid))
        arrived_vehicles += traci.simulation.getAllSubscriptionResults()[
            ''][0x79]  # throughput
    b = time.time()
    traci.close()
    avg_part_velocity = torch.tensor(part_velocity, dtype=torch.float).mean()

    avg_velocity = torch.tensor(total_velocity, dtype=torch.float).mean()
    avg_travel_time = torch.tensor(travel_time, dtype=torch.float).mean()
    avg_waiting_time = torch.tensor(waiting_time, dtype=torch.float).mean()
    print('======== arrived number:{} avg waiting time:{},avg velocity:{} avg_part_velocity: {} avg_travel_time: {}'.format(
        arrived_vehicles, avg_waiting_time, avg_velocity, avg_part_velocity, avg_travel_time))
    print("sim_time=", b-a)


def fixtime(flags, configs, sumoConfig):
    """
    固定时间控制模式 - 信号灯按照配置的固定相位时长运行，不使用RL控制
    
    此模式用于与强化学习控制进行对比，评估RL算法的效果。
    """
    from fixtime import city_fixtime_test
    
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, "--scale", configs['scale']]
    
    city_fixtime_test(flags, sumoCmd, configs)


def main(args):
    random_seed = 20000
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    flags = parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and flags.gpu == True else "cpu")
    # device = torch.device('cpu')
    print("Using device: {}".format(device))
    configs = EXP_CONFIGS
    configs['device'] = str(device)
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    configs['mode'] = flags.mode.lower()
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # 在测试模式和fixtime模式下，时间戳后面加上相应后缀，以便与训练时间戳区分
    if configs['mode'] == 'test':
        time_data += '_test'
    elif configs['mode'] == 'fixtime':
        time_data += '_fixtime'
    configs['time_data'] = str(time_data)
    if os.path.exists(os.path.join(os.path.dirname(__file__),'data')):
        if os.path.exists(os.path.join(os.path.dirname(__file__),'data',configs['mode']))==False:
            os.mkdir(os.path.join(os.path.dirname(__file__),'data',configs['mode']))
    configs['file_name'] = configs['time_data']
    # check the network
    configs['network'] = flags.network.lower()
    if configs['network'] == 'grid':
        from Network.grid import GridNetwork  # 切换network时使用这个（待修改）
        configs['grid_num'] = 5
        configs['scale'] = 1
        if configs['mode'] == 'simulate':
            configs['file_name'] = '{}x{}grid'.format(
                configs['grid_num'], configs['grid_num'])
        elif configs['mode'] == 'test':  # test
            configs['file_name'] = flags.replay_name.lower()
        # Generating Network
        network = GridNetwork(configs, grid_num=configs['grid_num'])
        network.generate_cfg(True, configs['mode'])
        NET_CONFIGS = network.get_configs()
        configs = merge_dict_non_conflict(configs, NET_CONFIGS)

    # Generating Network
    else:  # 从map文件加载
        print("Load from map file")
        from Network.map import MapNetwork
        # TODO Grid num需要删除
        configs['grid_num'] = 3
        configs['num_lanes'] = 2
        configs['load_file_name'] = configs['network']
        mapnet = MapNetwork(configs)
        MAP_CONFIGS = mapnet.get_tl_from_xml()

        for key in MAP_CONFIGS.keys():
            configs[key] = MAP_CONFIGS[key]

        mapnet.gen_net_from_xml()
        mapnet.gen_rou_from_xml()
        mapnet.generate_cfg(True, configs['mode'])
        mapnet._generate_add_xml()
        if configs['network'] == '3x3grid':
            configs['scale'] = str(1)
        if configs['network'] == '5x5grid':
            configs['scale'] = str(1)
        if configs['network'] == '5x5grid_v2':
            configs['scale'] = str(1.5)
        if configs['network'] == 'dunsan':
            configs['scale'] = str(1)
        if configs['network'] == 'dunsan_v2':
            configs['scale'] = str(0.8)
        print("Scale:",configs['scale'])

    # check the environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # check the mode
    if configs['mode'] == 'train':
        # init train setting
        configs['update_type'] = flags.update_type
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', time_data, 'net_data', configs['file_name']+'_train.sumocfg')
        train(flags, time_data, configs, sumoConfig)
    elif configs['mode'] == 'test':
        configs['file_name'] = flags.replay_name
        configs['replay_name'] = configs['time_data']
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', time_data, 'net_data', configs['time_data']+'_test.sumocfg')
        test(flags, configs, sumoConfig)
    elif configs['mode'] == 'fixtime':
        # 固定时间控制模式，需要加载SUPER_DQN_TRAFFIC_CONFIGS配置
        from configs import SUPER_DQN_TRAFFIC_CONFIGS
        configs = merge_dict_non_conflict(configs, SUPER_DQN_TRAFFIC_CONFIGS)
        # 确保num_agent从tl_rl_list计算
        if 'num_agent' not in configs and 'tl_rl_list' in configs:
            configs['num_agent'] = len(configs['tl_rl_list'])
        # fixtime模式使用file_name作为文件名，与生成的配置文件保持一致
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', time_data, 'net_data', configs['file_name']+'.sumocfg')
        fixtime(flags, configs, sumoConfig)
    else:  # simulate
        sumoConfig = os.path.join(
            configs['current_path'], 'Net_data', configs['file_name']+'_simulate.sumocfg')
        simulate(flags, configs, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])
