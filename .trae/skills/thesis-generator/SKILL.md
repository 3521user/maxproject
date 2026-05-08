---
name: "thesis-generator"
description: "Generates academic thesis in Chinese for traffic signal optimization using deep reinforcement learning. Invoke when user wants to create or regenerate a graduation thesis document."
---

# 基于深度强化学习的交通信号优化论文生成器

本技能用于生成完整的毕业论文文档（约22000字），包含以下五个章节：

## 论文结构

1. **第一章 绪论**
   - 研究背景与意义
   - 国内外研究现状
   - 论文的主要工作
   - 论文结构安排

2. **第二章 相关技术与理论基础**
   - 深度强化学习概述
   - Q-Learning与DQN算法原理
   - Double DQN算法原理
   - Dueling DQN算法原理
   - 优先级经验回放机制
   - 交通仿真平台SUMO介绍

3. **第三章 系统设计与实现**
   - 系统整体架构设计
   - 交通环境建模
   - 智能体神经网络设计
   - 奖励函数与动作空间设计
   - 训练策略与超参数设置

4. **第四章 实验与结果分析**
   - 实验环境与参数设置
   - Fixtime基线实验结果
   - Double DQN实验结果
   - Dueling DQN实验结果
   - 三种方法性能对比分析

5. **第五章 总结与展望**
   - 研究工作总结
   - 未来工作展望

## 实验数据

从tripinfo_stats.csv提取的关键性能指标：

| 方法 | 行程时间(秒) | 延误时间(秒) | 燃料消耗(L/100km) | CO2排放(g/km) |
|------|-------------|-------------|------------------|---------------|
| Fixtime | 877.94 | 735.93 | 333.72 | 7720.47 |
| Double DQN | 446.74 | 288.40 | 21.17 | 489.72 |
| Dueling DQN | 397.61 | 241.68 | 19.01 | 439.71 |

## 关键算法对比

1. **Fixtime（固定时间）**：基线方法，性能最差，按照预设固定时间切换信号灯
2. **Double DQN**：基础强化学习方法，使用双网络减少Q值过估计问题
3. **Dueling DQN**：主要工作，性能最优，改进点包括：
   - Dueling DQN架构：Q值分解为V(s)和A(s,a)
   - 优先级经验回放（PER）
   - Huber Loss替代MSE Loss
   - 奖励归一化
   - 学习率预热
   - 梯度裁剪

## 输出要求

生成docx格式文档，包含：
- 一级/二级/三级标题
- 文字段落（总字数约22000）
- 表格（性能对比）
- 公式（DQN/Dueling DQN公式）
- 适当的图片占位符