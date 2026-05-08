
class baseEnv():
    def __init__(self, configs):
        self.configs=configs
        '''
        base env
        '''

    def get_state(self):
        '''
        用于继承的函数
        return state torch.Tensor(dtype=torch.int)
        '''
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
    
    def collect_state(self):
        raise NotImplementedError

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        return reward torch.Tensor(dtype=torch.int)
        '''
        raise NotImplementedError

    def _toPhase(self, action):  # 将action转换为可解读的phase
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        signal_set = list()
        phase_set=tuple()
        phase = str()
        for _, a in enumerate(action):
            signal_set.append(self._getMovement(a))
        for j,signal in enumerate(signal_set):
            # 每个
            for i in range(4):  # 4车道
                phase = phase + 'g'+self.configs['numLane']*signal[j][2*i] + \
                    signal[j][2*i+1]+'r'  # 最后的r是u-turn
            phase_set+=phase
        print(phase_set)
        return phase_set

    def _toState(self, phase_set):  # 将env的phase转换为不可解读的state
        state_set=tuple()
        for i,phase in enumerate(phase_set):
            state = torch.zeros(8, dtype=torch.int)
            for i in range(4):  # 4车道
                phase = phase[1:]  # 右转
                state[i] = self._mappingMovement(phase[0])  # 提取直行信号
                phase = phase[3:]  # 直行
                state[i+1] = self._mappingMovement(phase[0])  # 提取左转信号
                phase = phase[1:]  # 左转
                phase = phase[1:]  # 掉头
            state_set+=state
        return state_set

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'G':
            return 1
        elif movement == 'r':
            return 0
        else:
            return -1  # error
