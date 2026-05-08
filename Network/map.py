
import os
import torch
from xml.etree.ElementTree import parse
from gen_net import Network
from configs import EXP_CONFIGS


class MapNetwork(Network):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_rl_list = list()
        self.offset_list = list()
        self.phase_list = list()
        self.common_phase = list()
        self.net_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.net.xml')
        self.rou_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.rou.xml')

    def get_tl_from_add_xml(self):
        add_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.add.xml')
        NET_CONFIGS = dict()
        NET_CONFIGS['phase_num_actions'] = {2: [[0, 0], [1, -1]],
                                            3: [[0, 0, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [-1, 1, 0]],
                                            4: [[1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 0],
                                                [-1, 0, 0, 1], [0, -1, 0, 1], [0, -1, 1, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [0, 0, -1, 1],
                                                [1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1],[1,-1,-1,1]],
                                            5: [[0, 0, 0, 0, 0]],
                                            6: [[0, 0, 0, 0, 0, 0]], }

        NET_CONFIGS['phase_type'] = list()

        NET_CONFIGS['rate_action_space'] = dict()
        for i in range(2, 7):  # 指定rate action_space
            NET_CONFIGS['rate_action_space'][i] = len(
                NET_CONFIGS['phase_num_actions'][i])

        NET_CONFIGS['tl_period'] = list()
        traffic_info = dict()
        print(add_file_path)
        add_net_tree = parse(add_file_path)
        tlLogicList = add_net_tree.findall('tlLogic')
        NET_CONFIGS['time_action_space'] = list()

        # 保存traffic info
        for tlLogic in tlLogicList:
            tl_id = tlLogic.attrib['id']
            traffic_info[tl_id] = dict()
            traffic_node_info = traffic_info[tl_id]
            traffic_node_info['min_phase'] = list()
            traffic_node_info['phase_duration'] = list()
            traffic_node_info['max_phase'] = list()
            traffic_node_info['min_phase'] = list()
            traffic_node_info['min_phase'] = list()

            # 整理rl agent数量
            self.tl_rl_list.append(tlLogic.attrib['id'])  # 添加要rl控制的tl_rl
            # 保存offset
            traffic_node_info['offset'] = int(tlLogic.attrib['offset'])
            self.offset_list.append(traffic_node_info['offset'])

            # 查找所有phase
            phaseList = tlLogic.findall('phase')
            phase_state_list = list()
            phase_duration_list = list()
            common_phase_list = list()
            phase_index_list = list()
            min_duration_list = list()
            max_duration_list = list()
            dif_max_list = list()
            dif_min_list = list()
            tl_period = 0  # phase set的总长度
            # 对于每个phase，找到长度等
            num_phase = 0  # phase数量过滤
            for i, phase in enumerate(phaseList):
                phase_state_list.append(phase.attrib['state'])
                phase_duration_list.append(int(phase.attrib['duration']))
                tl_period += int(phase.attrib['duration'])
                if int(phase.attrib['duration']) > 5:  # 视为Phase的数字
                    num_phase += 1
                    min_duration_list.append(int(phase.attrib['minDur']))
                    max_duration_list.append(int(phase.attrib['maxDur']))

                    dif_max_list.append(
                        (int(phase.attrib['maxDur'])-int(phase.attrib['duration']))/100.0)
                    dif_min_list.append(
                        (int(phase.attrib['duration'])-int(phase.attrib['minDur']))/100.0)
                    phase_index_list.append(i)
                    common_phase_list.append(int(phase.attrib['duration']))

            # 放入dictionary
            traffic_node_info['phase_list'] = phase_state_list
            traffic_node_info['phase_duration'] = phase_duration_list
            traffic_node_info['common_phase'] = common_phase_list
            traffic_node_info['phase_index'] = phase_index_list
            traffic_node_info['dif_min'] = dif_min_list
            traffic_node_info['dif_max'] = dif_max_list
            # 每个信号的长度
            traffic_node_info['period'] = tl_period
            NET_CONFIGS['tl_period'].append(tl_period)
            traffic_node_info['matrix_actions'] = NET_CONFIGS['phase_num_actions'][num_phase]
            traffic_node_info['min_phase'] = min_duration_list
            traffic_node_info['max_phase'] = max_duration_list
            traffic_node_info['num_phase'] = num_phase
            # 指定每个tl_rl的time_action_space
            # NET_CONFIGS['time_action_space'].append(abs(round((torch.min(torch.tensor(traffic_node_info['max_phase'])-torch.tensor(
            #     traffic_node_info['common_phase']), torch.tensor(traffic_node_info['common_phase'])-torch.tensor(traffic_node_info['min_phase']))/2).mean().item())))
            NET_CONFIGS['time_action_space'].append(4)  # 指定任意秒
            if 'grid' in self.configs['network']:
                NET_CONFIGS['phase_type'].append([0, 0])

            self.phase_list.append(phase_state_list)
            self.common_phase.append(phase_duration_list)
        if 'dunsan' in self.configs['network']:
            NET_CONFIGS['phase_type'] = [[0, 0], [0, 0], [0, 1], [
                1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [0, 1], [1, 0]]
        # TODO node interest pair计算器在network base中生成
        maximum = 0
        for key in traffic_info.keys():
            if maximum < len(traffic_info[key]['phase_duration']):
                maximum = len(traffic_info[key]['phase_duration'])
        NET_CONFIGS['max_phase_num'] = maximum

        # 用于road
        # 保存edge info
        self.configs['edge_info'] = list()
        edge_list = list()  # edge存在确认用
        net_tree = parse(self.net_file_path)
        edges = net_tree.findall('edge')
        for edge in edges:
            if 'function' not in edge.attrib.keys():
                edge_list.append({
                    'id': edge.attrib['id'],
                    'from': edge.attrib['from'],
                    'to': edge.attrib['to'],
                })
            self.configs['edge_info'].append(edge.attrib['id'])  # 保存所有edge
        # 保存node info
        self.configs['node_info'] = list()
        node_list = list()
        # interest list
        interest_list = list()
        # node interest pair
        node_interest_pair = dict()
        junctions = net_tree.findall('junction')
        # 决定state space size
        inflow_size = 0
        # 用于network
        for junction in junctions:
            node_id = junction.attrib['id']
            if junction.attrib['type'] == "traffic_light":  # 只分离正常node，红绿灯node
                node_list.append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })
                if node_id in self.tl_rl_list:  # 只保存学习的tl
                    i = 0
                    interests = list()
                    for edge in edge_list:
                        interest = dict()
                        if edge['to'] == node_id:  # inflow
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['inflow'] = edge['id']
                            for target_edge in edge_list:
                                if target_edge['from'] == edge['to'] and target_edge['to'] == edge['from']:
                                    interest['outflow'] = target_edge['id']
                                    break
                                else:
                                    interest['outflow'] = None

                            interests.append(interest)
                            i += 1  # 用于index标记

                        elif edge['from'] == node_id:
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['outflow'] = edge['id']
                            for target_edge in edge_list:
                                if target_edge['from'] == edge['to'] and target_edge['to'] == edge['from']:
                                    interest['inflow'] = target_edge['id']
                                    break
                                else:
                                    interest['inflow'] = None
                            interests.append(interest)
                            i += 1  # 用于index标记

                    # 确认是否存在重复后插入list
                    no_dup_outflow_list = list()
                    no_dup_interest_list = list()
                    for interest_comp in interests:
                        if interest_comp['outflow'] not in no_dup_outflow_list:
                            no_dup_outflow_list.append(
                                interest_comp['outflow'])
                            no_dup_interest_list.append(interest_comp)
                    interest_list.append(no_dup_interest_list)
                    node_interest_pair[node_id] = no_dup_interest_list
                    if inflow_size < len(no_dup_interest_list):
                        inflow_size = len(no_dup_interest_list)

            # 普通节点
            elif junction.attrib['type'] == "priority":  # 只分离正常node
                node_list.append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })
            else:
                pass
            self.configs['node_info'].append({
                'id': node_id,
                'type': junction.attrib['type'],
            })
            # 整理
        NET_CONFIGS['edge_info'] = self.configs['edge_info']
        NET_CONFIGS['node_info'] = self.configs['node_info']
        NET_CONFIGS['traffic_node_info'] = traffic_info
        NET_CONFIGS['interest_list'] = interest_list
        NET_CONFIGS['node_interest_pair'] = node_interest_pair
        NET_CONFIGS['tl_rl_list'] = self.tl_rl_list
        NET_CONFIGS['offset'] = self.offset_list
        NET_CONFIGS['phase_list'] = self.phase_list
        NET_CONFIGS['common_phase'] = self.common_phase
        NET_CONFIGS['state_space'] = inflow_size*2 + \
            2+2  # 左转,直行, 2是phase set形态, 2是phase dif dur(min max)
        # 添加num_agent配置
        NET_CONFIGS['num_agent'] = len(self.tl_rl_list)
        print("Agent Num:{}, Traffic Num:{}".format(
            len(self.tl_rl_list), len(node_list)))
        return NET_CONFIGS

    def get_tl_from_xml(self):
        if os.path.exists(os.path.join(self.configs['current_path'], 'Network', self.configs['load_file_name']+'.add.xml')):
            print("additional exists")
            return self.get_tl_from_add_xml()
        else:
            NET_CONFIGS = dict()
            NET_CONFIGS['phase_type'] = list()
            NET_CONFIGS['phase_num_actions'] = {2: [[0, 0], [1, -1], [-1, 1]],
                                                3: [[0, 0, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [-1, 1, 0]],
                                                4: [[1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 0],
                                                [-1, 0, 0, 1], [0, -1, 0, 1], [0, -1, 1, 0], [-1, 1, 0, 0], [-1, 0, 1, 0], [0, 0, -1, 1],
                                                [1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1],[1,-1,-1,1]],
                                                # 5: [[0, 0, 0, 0, 0]],
                                                # 6: [[0, 0, 0, 0, 0, 0]],
                                                }
            NET_CONFIGS['rate_action_space'] = dict()
            for i in NET_CONFIGS['phase_num_actions'].keys():  # 指定rate action_space
                NET_CONFIGS['rate_action_space'][i] = len(
                    NET_CONFIGS['phase_num_actions'][i])

            NET_CONFIGS['tl_period'] = list()
            traffic_info = dict()
            net_tree = parse(self.net_file_path)
            tlLogicList = net_tree.findall('tlLogic')
            NET_CONFIGS['time_action_space'] = list()
            if 'dunsan' == self.configs['network']:
                NET_CONFIGS['phase_type'] = [[0, 0], [0, 0], [0, 1], [
                    1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [0, 1], [1, 0]]
            # 保存traffic info
            for tlLogic in tlLogicList:
                if 'grid' in self.configs['network']:
                    NET_CONFIGS['phase_type'].append([0, 0])
                tl_id = tlLogic.attrib['id']
                traffic_info[tl_id] = dict()
                traffic_node_info = traffic_info[tl_id]
                traffic_node_info['min_phase'] = list()
                traffic_node_info['phase_duration'] = list()
                traffic_node_info['max_phase'] = list()
                traffic_node_info['min_phase'] = list()
                traffic_node_info['min_phase'] = list()

                # 整理rl agent数量
                self.tl_rl_list.append(tlLogic.attrib['id'])  # 添加要rl控制的tl_rl
                # 保存offset
                traffic_node_info['offset'] = int(tlLogic.attrib['offset'])
                self.offset_list.append(traffic_node_info['offset'])

                # 查找所有phase
                phaseList = tlLogic.findall('phase')
                phase_state_list = list()
                phase_duration_list = list()
                common_phase_list = list()
                phase_index_list = list()
                min_duration_list = list()
                max_duration_list = list()
                dif_min_list = list()
                dif_max_list = list()
                tl_period = 0  # phase set的总长度
                # 对于每个phase，找到长度等
                num_phase = 0  # phase数量过滤
                for i, phase in enumerate(phaseList):
                    phase_state_list.append(phase.attrib['state'])
                    this_phase_dur = phase.attrib['duration']
                    phase_duration_list.append(int(this_phase_dur))
                    tl_period += int(this_phase_dur)
                    # 视为Phase的数字
                    if int(this_phase_dur) > 5 and 'minDur' in phase.attrib.keys() and 'maxDur' in phase.attrib.keys():
                        num_phase += 1
                        min_duration_list.append(
                            int(phase.attrib['minDur']))
                        max_duration_list.append(
                            int(phase.attrib['maxDur']))
                        dif_max_list.append(
                            (int(phase.attrib['maxDur'])-int(phase.attrib['duration']))/100.0)
                        dif_min_list.append(
                            (int(phase.attrib['duration'])-int(phase.attrib['minDur']))/100.0)
                        phase_index_list.append(i)
                        common_phase_list.append(int(this_phase_dur))

                    elif int(this_phase_dur) > 5:
                        num_phase += 1
                        min_duration_list.append(
                            int(this_phase_dur)-5)
                        max_duration_list.append(
                            int(this_phase_dur)+5)
                        dif_max_list.append(5)
                        dif_min_list.append(5)
                        phase_index_list.append(i)
                        common_phase_list.append(int(this_phase_dur))

                # 放入dictionary
                traffic_node_info['phase_list'] = phase_state_list
                traffic_node_info['phase_duration'] = phase_duration_list
                traffic_node_info['common_phase'] = common_phase_list
                traffic_node_info['phase_index'] = phase_index_list
                traffic_node_info['dif_max'] = dif_max_list  # max dur的差异
                traffic_node_info['dif_min'] = dif_min_list  # min dur的差异
                # 每个信号的长度
                traffic_node_info['period'] = tl_period
                NET_CONFIGS['tl_period'].append(tl_period)
                traffic_node_info['matrix_actions'] = NET_CONFIGS['phase_num_actions'][num_phase]
                traffic_node_info['min_phase'] = min_duration_list
                traffic_node_info['max_phase'] = max_duration_list
                traffic_node_info['num_phase'] = num_phase
                # 指定每个tl_rl的time_action_space
                # NET_CONFIGS['time_action_space'].append(abs(round((torch.min(torch.tensor(traffic_node_info['max_phase'])-torch.tensor(
                #     traffic_node_info['common_phase']), torch.tensor(traffic_node_info['common_phase'])-torch.tensor(traffic_node_info['min_phase'])).float()).mean().item())))
                NET_CONFIGS['time_action_space'].append(4)

                self.phase_list.append(phase_state_list)
                self.common_phase.append(phase_duration_list)

            # TODO node interest pair计算器在network base中生成
            maximum = 0
            for key in traffic_info.keys():
                if maximum < len(traffic_info[key]['phase_duration']):
                    maximum = len(traffic_info[key]['phase_duration'])
            NET_CONFIGS['max_phase_num'] = maximum

            # 用于road
            # 保存edge info
            self.configs['edge_info'] = list()
            edges = net_tree.findall('edge')
            for edge in edges:
                if 'function' not in edge.attrib.keys():
                    self.configs['edge_info'].append({
                        'id': edge.attrib['id'],
                        'from': edge.attrib['from'],
                        'to': edge.attrib['to'],
                    })
            # 保存node info
            self.configs['node_info'] = list()
            node_list = list()
            # interest list
            interest_list = list()
            # node interest pair
            node_interest_pair = dict()
            junctions = net_tree.findall('junction')
            # 决定state space size
            inflow_size = 0
            # 用于network
            for junction in junctions:
                node_id = junction.attrib['id']
                if junction.attrib['type'] == "traffic_light":  # 只分离正常node，红绿灯node
                    node_list.append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                    # node决定完成
                    # edge是?
                    i = 0
                    interests = list()
                    for edge in self.configs['edge_info']:
                        interest = dict()
                        if edge['to'] == node_id:  # inflow
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['inflow'] = edge['id']
                            for tmpEdge in self.configs['edge_info']:  # outflow
                                if tmpEdge['from'] == node_id and edge['from'] == tmpEdge['to']:
                                    interest['outflow'] = tmpEdge['id']
                                    break
                                else:
                                    interest['outflow'] = None
                            # tmp_edge=str(-int(edge['id']))
                            # if tmp_edge in edge_list:
                            #     interest['outflow']=tmp_edge
                            # else:
                            #     interest['outflow']=None
                            interests.append(interest)
                            i += 1  # 用于index标记

                        elif edge['from'] == node_id:
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['outflow'] = edge['id']
                            for tmpEdge in self.configs['edge_info']:  # outflow
                                if tmpEdge['to'] == node_id and edge['to'] == tmpEdge['from']:
                                    interest['inflow'] = tmpEdge['id']
                                    break
                                else:
                                    interest['inflow'] = None
                            # tmp_edge=str(-int(edge['id']))
                            # if tmp_edge in edge_list:
                            #     interest['inflow']=tmp_edge
                            # else:
                            #     interest['inflow']=None
                            interests.append(interest)
                            i += 1  # 用于index标记

                    # 确认是否存在重复后插入list
                    no_dup_outflow_list = list()
                    no_dup_interest_list = list()
                    for interest_comp in interests:
                        if interest_comp['outflow'] not in no_dup_outflow_list:
                            no_dup_outflow_list.append(
                                interest_comp['outflow'])
                            no_dup_interest_list.append(interest_comp)
                    interest_list.append(no_dup_interest_list)
                    node_interest_pair[node_id] = no_dup_interest_list
                    if inflow_size < len(no_dup_interest_list):
                        inflow_size = len(no_dup_interest_list)

                # 普通节点
                elif junction.attrib['type'] == "priority":  # 只分离正常node
                    node_list.append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                else:
                    pass
                self.configs['node_info'].append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })

            # 整理
            NET_CONFIGS['node_info'] = self.configs['node_info']
            NET_CONFIGS['edge_info'] = self.configs['edge_info']

            NET_CONFIGS['traffic_node_info'] = traffic_info
            NET_CONFIGS['interest_list'] = interest_list
            NET_CONFIGS['node_interest_pair'] = node_interest_pair
            NET_CONFIGS['tl_rl_list'] = self.tl_rl_list
            NET_CONFIGS['offset'] = self.offset_list
            NET_CONFIGS['phase_list'] = self.phase_list
            NET_CONFIGS['common_phase'] = self.common_phase
            # 左转,直行 , 2个phase type(one hot), 2个phaseduration(min max)
            NET_CONFIGS['state_space'] = inflow_size*2+2+2
            # 添加num_agent配置
            NET_CONFIGS['num_agent'] = len(self.tl_rl_list)

            return NET_CONFIGS

    def gen_net_from_xml(self):
        net_tree = parse(self.net_file_path)
        if self.configs['mode'] == 'train' or self.configs['mode'] == 'test' or self.configs['mode'] == 'fixtime':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                             self.configs['time_data'], 'net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
        else:  # simulate
            gen_file_name = str(os.path.join(
                self.configs['current_path'], 'Net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)

    def gen_rou_from_xml(self):
        net_tree = parse(self.rou_file_path)
        if self.configs['mode'] == 'train' or self.configs['mode'] == 'test' or self.configs['mode'] == 'fixtime':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                             self.configs['time_data'], 'net_data', self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
        else:
            gen_file_name = str(os.path.join(self.configs['current_path'], 'Net_data',
                                             self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
