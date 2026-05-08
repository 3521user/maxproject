import torch
Edges = list()
Nodes = list()
Vehicles = list()
EXP_CONFIGS = {
    'num_lanes': 3,
    'model': 'normal',
    'file_name': '3x3grid',
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 3600,
    'sim_start': 0,
    'max_steps': 3600,
    'num_epochs': 1000,
    'edge_info': Edges,
    'node_info': Nodes,
    'vehicle_info': Vehicles,
    'mode': 'simulate',
}
# city DQN

# Decentralized_DQN
SUPER_DQN_TRAFFIC_CONFIGS = {
    # 按1,agent,num_phase顺序排列
    'min_phase': [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]],
    'common_phase': [[37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37], [37, 37, 37, 37]],
    # commonphase后面始终加上3秒，完成最后一个元素后加上3秒yellow变成tl_period
    # 按1,agent,num_phase顺序排列
    'max_phase': [[49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49]],
    # 按1, agent顺序排列
    'phase_period': [[160], [160], [160], [160], [160], [160], [160], [160], [160]],
    'matrix_actions': [[0, 0, 0, 0], [1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1],
                       [1, 0, 0, -1], [1, 0, -1, 0], [1, 0, 0, -1], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1]]
}
