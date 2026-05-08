# 基于深度强化学习的交通信号优化设计与实现
##### Design and Implementation of Traffic Signal Optimization Based on Deep Reinforcement Learning

针对传统交通信号控制适配性弱、多路口协同不足的问题，结合 SUMO 仿真与 DQN 算法，构建去中心化多智能体信号控制方案，依托局部感知与信息共享实现协同调度。该研究可完善强化学习落地路径，为智能交通信号优化提供理论参考与实用技术支撑，有效改善交通通行状况。

### 使用方法

```shell script
   python run.py train --network 5x5grid
```

```shell script
   python run.py test --network 5x5grid --replay_name 04-28_11-22-02 --replay_epoch 600
```

```shell script
   python run.py test --network 5x5grid --replay_name 04-25_18-50-01 --replay_epoch 550
```

```shell script
   python run.py fixtime --network 5x5grid
```

- 查看结果
Tensorboard
```shell script
    tensorboard --logdir ./training_data
```
超参数保存在json文件中，模型保存在 `./training_data/[运行时间]/model` 目录下。

- 回放模型
```shell script
    python run.py test --replay_name /training_data目录中的/replay_data/ --replay_epoch NUM
```
