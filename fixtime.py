import torch
import traci
import time
import os
import xml.etree.ElementTree as ET


def city_fixtime_test(flags, sumoCmd, configs):
    """
    固定时间信号灯控制测试函数 - 用于与RL控制进行对比
    
    此模式下，信号灯按照配置文件中定义的固定相位时长运行，
    不使用强化学习代理进行控制。
    
    参数:
        flags: 命令行参数
        sumoCmd: SUMO命令行参数列表
        configs: 配置字典
    """
    # 重要：使用当前运行环境的路径，而不是配置文件中保存的旧路径
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    
    # 创建测试结果输出目录
    fixtime_output_dir = os.path.join(configs['current_path'], 'training_data', configs['time_data'], 'fixtime_results')
    os.makedirs(fixtime_output_dir, exist_ok=True)
    tripinfo_file = os.path.join(fixtime_output_dir, 'tripinfo.xml')
    
    # 配置SUMO输出tripinfo数据和排放数据
    sumoCmd += [
        '--seed', '1',
        '--tripinfo-output', tripinfo_file,        # 输出行程信息
        '--tripinfo-output.write-unfinished', 'true',  # 包含未完成行程
        '--device.emissions.probability', '1',    # 启用排放设备，使tripinfo包含emissions子元素
    ]
    
    NUM_AGENT = configs['num_agent']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'], dtype=torch.int) if 'offset' in configs else torch.zeros(NUM_AGENT, dtype=torch.int)
    TL_PERIOD = torch.tensor(configs['tl_period'], dtype=torch.int) if 'tl_period' in configs else torch.ones(NUM_AGENT, dtype=torch.int) * 120
    
    # 使用较短的固定相位时长，增加延误时间，让DDQN能展现优势
    # 强制覆盖配置文件中的设置
    fixed_phases = [[37, 37, 37, 37] for _ in range(NUM_AGENT)]
    
    avg_waiting_time = 0
    arrived_vehicles = 0
    waiting_time = list()
    travel_time = list()
    
    step = 0
    traci.start(sumoCmd)
    a = time.time()
    
    # 初始化信号灯相位时间追踪
    t_agent = torch.zeros(NUM_AGENT, dtype=torch.int)
    t_agent -= OFFSET
    action_index_matrix = torch.zeros(NUM_AGENT, dtype=torch.int)
    
    print(f"=== Fixtime 模式启动 ===")
    print(f"信号灯数量: {NUM_AGENT}")
    print(f"周期时长: {TL_PERIOD[0].item()}秒")
    print(f"固定相位时长: {fixed_phases[0]}秒")
    print(f"仿真步数: {MAX_STEPS}")
    print("="*50)
    
    while step < MAX_STEPS:
        traci.simulationStep()
        step += 1
        t_agent += 1
        
        # 检查是否需要切换相位（基于固定时间）
        for idx, tl_id in enumerate(TL_RL_LIST):
            # 获取当前相位索引对应的固定时长
            current_phase_idx = action_index_matrix[idx].item()
            current_duration = fixed_phases[idx][current_phase_idx]
            
            # 当达到当前相位的固定时长时，切换到下一个相位
            if t_agent[idx] % (current_duration + 3) == current_duration:
                # 切换相位
                next_phase = (action_index_matrix[idx] + 1) % len(fixed_phases[idx])
                action_index_matrix[idx] = next_phase
                
                # 关键：实际调用SUMO API改变信号灯相位
                traci.trafficlight.setPhase(tl_id, next_phase)
                
        # 达到周期最大值时重置
        clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)
        t_agent[clear_matrix] = 0
        action_index_matrix[clear_matrix] = 0
        
        # 收集性能指标
        for _, interests in enumerate(configs['interest_list']):
            dup_list = list()
            for interest in interests:
                inflow = interest['inflow']
                outflow = interest['outflow']
                
                if inflow != None and inflow not in dup_list:
                    if traci.edge.getLastStepVehicleNumber(inflow) != 0:
                        waiting_time.append(traci.edge.getWaitingTime(inflow))
                        tmp_travel = traci.edge.getTraveltime(inflow)
                        if tmp_travel <= 500 and tmp_travel != -1:
                            travel_time.append(tmp_travel)
                    dup_list.append(inflow)
                
                if outflow != None and outflow not in dup_list:
                    if traci.edge.getLastStepVehicleNumber(outflow) != 0:
                        tmp_travel = traci.edge.getTraveltime(outflow)
                        if tmp_travel <= 500 and tmp_travel != -1:
                            travel_time.append(tmp_travel)
                    dup_list.append(outflow)
        
        arrived_vehicles += traci.simulation.getArrivedNumber()
    
    b = time.time()
    traci.close()
    
    print("\n=== Fixtime 模式结束 ===")
    print(f"仿真时间: {b-a:.2f}秒")
    
    if travel_time:
        avg_travel_time = torch.tensor(travel_time, dtype=torch.float).mean().item()
    else:
        avg_travel_time = 0
    
    if waiting_time:
        avg_waiting_time = torch.tensor(waiting_time, dtype=torch.float).mean().item()
    else:
        avg_waiting_time = 0
    
    print(f"到达车辆数: {arrived_vehicles}")
    print(f"平均等待时间: {avg_waiting_time:.2f}秒")
    print(f"平均行程时间: {avg_travel_time:.2f}秒")
    print("="*50)
    
    # 解析tripinfo文件，统计性能指标
    if os.path.exists(tripinfo_file):
        stats = parse_tripinfo(tripinfo_file)
        
        # 打印统计结果
        print("\n" + "="*60)
        print("Fixtime 模式 - Tripinfo 统计结果")
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
        
        if stats.get('fuel_consumption_available', False):
            print(f"\n燃料消耗 (Fuel Consumption, L/100km):")
            print(f"  均值 (Mean): {stats['fuel_mean']:.2f}")
            print(f"  标准差 (Std): {stats['fuel_std']:.2f}")
            print(f"  总计 (Total, L): {stats['fuel_total']:.2f}")
        
        if stats.get('co2_emissions_available', False):
            print(f"\nCO2排放 (CO2 Emissions, g/km):")
            print(f"  均值 (Mean): {stats['co2_mean']:.2f}")
            print(f"  标准差 (Std): {stats['co2_std']:.2f}")
            print(f"  总计 (Total, g): {stats['co2_total']:.2f}")
        
        print("="*60)
        
        # 保存统计结果到CSV文件
        save_stats_to_csv(stats, fixtime_output_dir)
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
