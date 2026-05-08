import torch
import traci
import time
import os
import xml.etree.ElementTree as ET
from utils import load_params
from Agent.base import merge_dict_non_conflict


def city_dqn_test(flags, sumoCmd, configs):
    """
    DQN测试函数 - 用于评估训练好的模型性能
    """
    from Agent.super_dqn import Trainer
    from Env.CityEnv import CityEnv

    # 加载训练好的模型配置
    if flags.replay_name is not None:
        configs = load_params(configs, flags.replay_name)
        configs['replay_epoch'] = str(flags.replay_epoch)
        configs['mode'] = 'test'

    # 重要：使用当前运行环境的路径，而不是配置文件中保存的旧路径
    # 这样可以确保在不同电脑上运行时使用正确的路径
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))

    # 初始化各交叉口相位矩阵
    phase_num_matrix = torch.tensor(
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])

    sumoCmd += ['--seed', '1']

    # 创建测试结果输出目录
    tripinfo_output_dir = os.path.join(configs['current_path'], 'training_data', configs['time_data'], 'test_results')
    os.makedirs(tripinfo_output_dir, exist_ok=True)
    tripinfo_file = os.path.join(tripinfo_output_dir, 'tripinfo.xml')
    emission_file = os.path.join(tripinfo_output_dir, 'emission.xml')

    # 配置SUMO输出tripinfo和emission数据
    sumoCmd += [
        '--tripinfo-output', tripinfo_file,        # 输出行程信息
        '--tripinfo-output.write-unfinished', 'true',  # 包含未完成行程
        '--device.emissions.probability', '1',    # 启用排放设备，使tripinfo包含emissions子元素
    ]

    agent = Trainer(configs)
    agent.save_params(configs['time_data'])
    agent.load_weights(flags.replay_name)

    NUM_AGENT = configs['num_agent']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],
                          device=configs['device'], dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=configs['device'], dtype=torch.int)

    avg_waiting_time = 0
    avg_part_velocity = 0
    avg_velocity = 0
    arrived_vehicles = 0
    part_velocity = list()
    waiting_time = list()
    total_velocity = list()
    travel_time = list()

    with torch.no_grad():
        step = 0
        traci.start(sumoCmd)
        env = CityEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=configs['device'])
        # Mask矩阵：当TL_Period结束时为True
        mask_matrix = torch.ones(
            (NUM_AGENT), dtype=torch.bool, device=configs['device'])

        # 仅在MAX Period内递增的t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=configs['device'])
        t_agent -= OFFSET

        # Action矩阵：比较相同时收集state，不存在的state用zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=configs['device'])  # 需要处理黄灯3秒
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=configs['device'])  # 当前是第几个相位
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
            if mask_matrix.sum()>0:
                print(actions.transpose(1,2))
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
            # 如果需要切换action，则增加action索引（通过tensor切片）
            for idx,_ in enumerate(TL_RL_LIST):
                action_update_mask[idx] = torch.eq(  # update只需要接收真实相位来决定
                    t_agent[idx], action_matrix[idx, action_index_matrix[idx]].view(-1))  # 使用0是因为索引问题

            action_index_matrix[action_update_mask] += 1
            # 如果超过agent的最大phase，则将该agent的action索引重置为0
            action_index_matrix[clear_matrix] = 0

            # 更新mask，矩阵转换为True
            t_agent[clear_matrix] = 0
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False


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
                            waiting_time.append(traci.edge.getWaitingTime(inflow))
                            tmp_travel = traci.edge.getTraveltime(inflow)
                            if tmp_travel<=500 and tmp_travel !=-1:  # 过滤异常值
                                travel_time.append(tmp_travel)
                        dup_list.append(inflow)

                    if outflow != None and outflow not in dup_list:
                        if traci.edge.getLastStepVehicleNumber(outflow) != 0:
                            tmp_travel = traci.edge.getTraveltime(outflow)
                            if tmp_travel<=500 and tmp_travel !=-1:  # 过滤异常值
                                travel_time.append(tmp_travel)
                        dup_list.append(interest['outflow'])

            next_state = env.collect_state(
                action_update_mask, action_index_matrix, mask_matrix)
            state = next_state
            # info

            arrived_vehicles += traci.simulation.getArrivedNumber()


        b = time.time()
        traci.close()
        print("time:", b-a)
        avg_travel_time = torch.tensor(travel_time, dtype=torch.float).mean()
        avg_waiting_time = torch.tensor(waiting_time, dtype=torch.float).mean()
        print('======== arrived number:{} avg waiting time:{} avg_travel_time: {}'.format(
            arrived_vehicles, avg_waiting_time, avg_travel_time))
        print("Reward: {}".format(env.cum_reward.sum()))

        # 解析tripinfo文件，统计性能指标
        print(f"\n[DEBUG] Checking tripinfo file: {tripinfo_file}")
        print(f"[DEBUG] File exists: {os.path.exists(tripinfo_file)}")
        print(f"[DEBUG] Output directory: {tripinfo_output_dir}")

        if os.path.exists(tripinfo_file):
            stats = parse_tripinfo(tripinfo_file)

            # 打印统计结果
            print("\n" + "="*60)
            print("Tripinfo 统计结果 (Tripinfo Statistics):")
            print("="*60)
            print(f"总车辆数 (Total Vehicles): {stats['total_vehicles']}")
            print(f"已完成行程车辆 (Completed): {stats['completed_vehicles']}")
            print(f"未完成行程车辆 (Unfinished): {stats['unfinished_vehicles']}")
            print(f"\n行程时间 (Travel Time, s):")
            print(f"  均值 (Mean): {stats['travel_time_mean']:.2f}")
            print(f"  标准差 (Std): {stats['travel_time_std']:.2f}")
            print(f"  最小值 (Min): {stats['travel_time_min']:.2f}")
            print(f"  最大值 (Max): {stats['travel_time_max']:.2f}")
            print(f"\n延误时间 (Delay, s):")
            print(f"  均值 (Mean): {stats['delay_mean']:.2f}")
            print(f"  标准差 (Std): {stats['delay_std']:.2f}")
            print(f"  最小值 (Min): {stats['delay_min']:.2f}")
            print(f"  最大值 (Max): {stats['delay_max']:.2f}")

            # 如果有燃料消耗数据，则打印
            if stats.get('fuel_consumption_available', False):
                print(f"\n燃料消耗 (Fuel Consumption, L/100km):")
                print(f"  均值 (Mean): {stats['fuel_mean']:.2f}")
                print(f"  标准差 (Std): {stats['fuel_std']:.2f}")
                print(f"  总计 (Total, L): {stats['fuel_total']:.2f}")

            # 如果有CO2排放数据，则打印
            if stats.get('co2_emissions_available', False):
                print(f"\nCO2排放 (CO2 Emissions, g/km):")
                print(f"  均值 (Mean): {stats['co2_mean']:.2f}")
                print(f"  标准差 (Std): {stats['co2_std']:.2f}")
                print(f"  总计 (Total, g): {stats['co2_total']:.2f}")

            print("="*60)

            # 保存统计结果到CSV文件
            save_stats_to_csv(stats, tripinfo_output_dir)
        else:
            print(f"警告 (Warning): Tripinfo文件未找到 at {tripinfo_file}")


def parse_tripinfo(tripinfo_file):
    """
    解析tripinfo.xml文件，统计各项指标

    主要指标:
    - Travel Time (行程时间): 车辆从起点到终点的时间
    - Delay (延误时间): 车辆因交通拥堵或信号灯等待而额外花费的时间
    - Fuel (燃料消耗): 每百公里消耗的燃油量 (L/100km)
    - CO2 (碳排放): 每公里排放的二氧化碳量 (g/km)

    SUMO原生单位:
    - fuel_abs: mg (毫克)
    - CO2_abs: mg (毫克)
    - routeLength: m (米)

    Returns:
        dict: 包含各项统计指标的字典
    """
    stats = {
        'total_vehicles': 0,
        'completed_vehicles': 0,
        'unfinished_vehicles': 0,
        'travel_time': [],
        'delay': [],
        'fuel': [],
        'co2': [],
        'fuel_consumption_available': False,
        'co2_emissions_available': False
    }

    # 汽油密度约为0.75 g/mL
    FUEL_DENSITY_G_PER_ML = 0.75

    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()

        for tripinfo in root.findall('tripinfo'):
            stats['total_vehicles'] += 1

            duration = float(tripinfo.get('duration', 0))
            stats['travel_time'].append(duration)

            waiting_time = float(tripinfo.get('waitingTime', 0))
            stats['delay'].append(waiting_time)

            # 获取行程距离（米）
            route_length = float(tripinfo.get('routeLength', 0))

            # 获取emissions子元素
            emissions = tripinfo.find('emissions')
            if emissions is not None:
                # fuel_abs单位是mg（毫克），需要转换为L/100km
                # 换算步骤:
                # 1. mg -> g: /1000
                # 2. g -> mL: /0.75 (汽油密度)
                # 3. mL -> L: /1000
                # 4. L/100km = fuel_L / (distance_km) * 100
                fuel = emissions.get('fuel_abs')
                if fuel is not None and route_length > 0:
                    stats['fuel_consumption_available'] = True
                    try:
                        fuel_mg = float(fuel)
                        # mg -> g -> mL -> L
                        fuel_L = fuel_mg / 1000 / FUEL_DENSITY_G_PER_ML / 1000
                        distance_km = route_length / 1000
                        fuel_per_100km = fuel_L / distance_km * 100
                        stats['fuel'].append(fuel_per_100km)
                    except ValueError:
                        pass

                # CO2_abs单位是mg(毫克)
                # g/km = CO2_mg / 1000 / distance_km = CO2_mg / distance_m
                co2 = emissions.get('CO2_abs')
                if co2 is not None and route_length > 0:
                    stats['co2_emissions_available'] = True
                    try:
                        co2_mg = float(co2)
                        co2_per_km = co2_mg / 1000 / (route_length / 1000)
                        stats['co2'].append(co2_per_km)
                    except ValueError:
                        pass

            if duration > 0:
                stats['completed_vehicles'] += 1
            else:
                stats['unfinished_vehicles'] += 1

        travel_time_arr = torch.tensor(stats['travel_time'], dtype=torch.float) if stats['travel_time'] else torch.tensor([0.0])
        delay_arr = torch.tensor(stats['delay'], dtype=torch.float) if stats['delay'] else torch.tensor([0.0])

        stats['travel_time_mean'] = travel_time_arr.mean().item() if len(travel_time_arr) > 0 else 0.0
        stats['travel_time_std'] = travel_time_arr.std().item() if len(travel_time_arr) > 1 else 0.0
        stats['travel_time_min'] = travel_time_arr.min().item() if len(travel_time_arr) > 0 else 0.0
        stats['travel_time_max'] = travel_time_arr.max().item() if len(travel_time_arr) > 0 else 0.0

        stats['delay_mean'] = delay_arr.mean().item() if len(delay_arr) > 0 else 0.0
        stats['delay_std'] = delay_arr.std().item() if len(delay_arr) > 1 else 0.0
        stats['delay_min'] = delay_arr.min().item() if len(delay_arr) > 0 else 0.0
        stats['delay_max'] = delay_arr.max().item() if len(delay_arr) > 0 else 0.0

        if stats['fuel']:
            fuel_arr = torch.tensor(stats['fuel'], dtype=torch.float)
            stats['fuel_mean'] = fuel_arr.mean().item()
            stats['fuel_std'] = fuel_arr.std().item() if len(fuel_arr) > 1 else 0.0
            stats['fuel_total'] = fuel_arr.sum().item()
        else:
            stats['fuel_mean'] = 0.0
            stats['fuel_std'] = 0.0
            stats['fuel_total'] = 0.0

        if stats['co2']:
            co2_arr = torch.tensor(stats['co2'], dtype=torch.float)
            stats['co2_mean'] = co2_arr.mean().item()
            stats['co2_std'] = co2_arr.std().item() if len(co2_arr) > 1 else 0.0
            stats['co2_total'] = co2_arr.sum().item()
        else:
            stats['co2_mean'] = 0.0
            stats['co2_std'] = 0.0
            stats['co2_total'] = 0.0

    except Exception as e:
        print(f"Error parsing tripinfo file: {e}")
        import traceback
        traceback.print_exc()

    return stats


def save_stats_to_csv(stats, output_dir):
    """
    将统计结果保存为CSV文件

    输出文件包含:
    - 车辆统计 (总数、完成数、未完成数)
    - 行程时间统计 (均值、标准差、最小值、最大值)
    - 延误时间统计 (均值、标准差、最小值、最大值)
    - 燃料消耗统计 (均值、标准差、总计)
    - CO2排放统计 (均值、标准差、总计)

    Args:
        stats: 统计指标字典
        output_dir: 输出目录路径
    """
    import csv

    csv_file = os.path.join(output_dir, 'tripinfo_stats.csv')

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Vehicles', stats['total_vehicles']])
        writer.writerow(['Completed Vehicles', stats['completed_vehicles']])
        writer.writerow(['Unfinished Vehicles', stats['unfinished_vehicles']])
        writer.writerow([])
        writer.writerow(['Travel Time (s)', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerow(['', stats['travel_time_mean'], stats['travel_time_std'],
                        stats['travel_time_min'], stats['travel_time_max']])
        writer.writerow([])
        writer.writerow(['Delay (s)', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerow(['', stats['delay_mean'], stats['delay_std'],
                        stats['delay_min'], stats['delay_max']])

        if stats.get('fuel_consumption_available', False):
            writer.writerow([])
            writer.writerow(['Fuel (L/100km)', 'Mean', 'Std', 'Total (L)'])
            writer.writerow(['', stats['fuel_mean'], stats['fuel_std'], stats['fuel_total']])

        if stats.get('co2_emissions_available', False):
            writer.writerow([])
            writer.writerow(['CO2 (g/km)', 'Mean', 'Std', 'Total (g)'])
            writer.writerow(['', stats['co2_mean'], stats['co2_std'], stats['co2_total']])

    print(f"\nStatistics saved to: {csv_file}")
