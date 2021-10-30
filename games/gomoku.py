import datetime
import math
import os

import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # 这里有更多信息: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # numpy、torch和游戏的种子
        self.max_num_gpus = None  # 确定要使用的最大GPU数。如果有足够的显存，使用单个GPU（将其设置为1）通常会更快。None会使用所有可用的GPU


        ### 游戏
        self.observation_shape = (3, 11, 11)  # 游戏观察的尺寸必须为3（通道、高度、宽度）。对于1D数组，请将其形状改为（1，1，数组长度）
        self.action_space = list(range(11 * 11))  # 所有可能动作的固定列表。您应该只编辑长度
        self.players = list(range(2))  # 玩家的列表。您应该只编辑长度
        self.stacked_observations = 0  # 要添加到 当前观察 的 先前观察 和 先前动作 的数量

        # 评估
        self.muzero_player = 0  # Muzero 玩游戏的先后手（0:Muzero 先手，1:Muzero 后手）
        self.opponent = "random"  # MuZero 对战的硬编码智能体，以评估他在多人游戏中的进展。它不影响训练。如果在游戏类中实现，分为 None, "random" or "expert"

        ### 自博弈
        self.num_workers = 2  # 给回放池提供数据的并发自博弈 工作进程/线程 数目
        self.selfplay_on_gpu = False
        self.max_moves = 121  # 如果游戏未在之前完成，则能移动的最大数量
        self.num_simulations = 400  # 自模拟的未来移动次数（MCTS对当前根结点展开模拟的次数）
        self.discount = 1  # 按时间顺序排列的奖励折扣
        self.temperature_threshold = None  # 将 visit_softmax_temperature_fn 给出的温度降至0（即选择最佳动作）之前的移动次数。如果是 None，visit_softmax_temperature_fn 每次都会被用到

        # 根先验探索噪声
        self.root_dirichlet_alpha = 0.3  # 狄利克雷分布的α参数
        self.root_exploration_fraction = 0.25  # 噪声所占的比例

        # UCB 公式
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### 神经网络
        self.network = "resnet"  # "resnet" 或 "fullyconnected"
        self.support_size = 10  # 价值和奖励被缩放（几乎都是用sqrt）并编码在一个向量上，向量的范围为-support_size到support_size。选择它以便使 support_size <= sqrt(max(abs(discounted reward)))
        
        # 残差网络
        self.downsample = False  # 表示网络 前的下采样观测值, False / "CNN" (更轻量级的) / "resnet" (见论文附录网络架构)
        self.blocks = 6  # ResNet中的块数
        self.channels = 128  # ResNet中的通道数
        self.reduced_channels_reward = 2  # 奖励头 中的通道数
        self.reduced_channels_value = 2  # 价值头 中的通道数
        self.reduced_channels_policy = 4  # 策略头 中的通道数
        self.resnet_fc_reward_layers = [64]  # 在 动态网络 的 奖励头 中定义隐藏层
        self.resnet_fc_value_layers = [64]  # 在 预测网络 的 价值头 中定义隐藏层
        self.resnet_fc_policy_layers = [64]  # 在 预测网络 的 策略头 中定义隐藏层
        
        # 全连接神经网络
        self.encoding_size = 32
        self.fc_representation_layers = []  # 定义 表示网络 中的隐藏层
        self.fc_dynamics_layers = [64]  # 定义 动态网络 中的隐藏层
        self.fc_reward_layers = [64]  # 定义 奖励网络 中的隐藏层
        self.fc_value_layers = []  # 定义 价值网络 中的隐藏层
        self.fc_policy_layers = []  # 定义 策略网络 中的隐藏层



        ### 训练
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # 存储 模型权重 和 Tensorboard 日志的路径
        self.save_model = True  # 将 results_path 中的检查点另存为 model.checkpoint
        self.training_steps = 10000  # 训练步总数（即根据批次更新权重）
        self.batch_size = 512  # 在每个训练步中 训练的游戏片段 的数量
        self.checkpoint_interval = 50  # 保存模型的训练步间隔
        self.value_loss_weight = 1  # 缩放价值损失以避免价值函数的过拟合，论文建议为0.25（见论文附录 重新分析）
        self.train_on_gpu = torch.cuda.is_available()  # 在GPU上训练（如果可用）

        self.optimizer = "Adam"  # "Adam" or "SGD". 论文中使用 SGD
        self.weight_decay = 1e-4  # L2权重正则化
        self.momentum = 0.9  # 仅当优化器为SGD时使用

        # 指数学习速率调度器
        self.lr_init = 0.002  # 初始学习率
        self.lr_decay_rate = 0.9  # 将其设置为1以使用恒定的学习速率
        self.lr_decay_steps = 10000



        ### 回放池
        self.replay_buffer_size = 10000  # 要保留在回放池中的自博弈游戏盘数
        self.num_unroll_steps = 121  # 每盘游戏为批元素保留的游戏移动数（移动为样本的单位），即论文中的“展开的K假设步”
        self.td_steps = 121  # 计算 目标价值 时要考虑的未来步数，即论文中的“n-step前瞻的n”
        self.PER = True  # 优先回放（见论文附录 训练），优先选择回放池中网络意外的元素
        self.PER_alpha = 0.5  # 使用多少优先级，0对应于uniform分布的情况，论文建议1

        # 重新分析（见论文附录 重新分析）
        self.use_last_model_value = False  # 使用最后一个模型提供更新鲜、稳定的 n-step 价值（见论文附录 重新分析）
        self.reanalyse_on_gpu = False



        ### 调整 自博弈/训练 比率，以避免 过拟合/欠拟合
        self.self_play_delay = 0  # 每次玩一盘游戏后等待的秒数
        self.training_delay = 0  # 每个训练步后等待的秒数
        self.ratio = 1  # 每个 自博弈步 的 期望训练步 比率。与同步版本等效，训练可能需要更长的时间。设置为“None”以禁用它


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        更改 访问次数分布 的参数——确保随着训练的进行，动作选择变得更加贪婪。越小，越有可能选择最佳行动（即访问次数最多）。

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    游戏装饰器
    """

    def __init__(self, seed=None):
        self.env = Gomoku()

    def step(self, action):
        """
        在游戏中执行一个动作
        
        Args:
            action : 要采取的 动作空间 中的 动作。

        Returns:
            新的观察、奖励和（标记游戏是否结束）布尔值。
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        返回当前的玩家

        Returns:
            当前玩家，它应该是配置中玩家列表的一个元素
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        应在每个回合返回合法动作，如果不可用，可返回整个动作空间。在每个回合，游戏必须能够处理一个返回的动作。
        对于计算合法动作太长的复杂游戏，其想法是将合法动作定义为整个动作空间，但如果动作非法，则返回负奖励。

        Returns:
            An array of integers，动作空间的子集
        """
        return self.env.legal_actions()

    def reset(self):
        """
        为新游戏重置游戏
        
        Returns:
            游戏的初始化观察
        """
        return self.env.reset()

    def close(self):
        """
        正确地结束游戏
        """
        pass

    def render(self):
        """
        可视化（渲染）游戏观察
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        对于多人游戏，要求用户采取合法动作并返回相应的动作编号

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        将动作编号转换为表示该动作的字符串
        Args:
            action_number: an integer from the action space.
        Returns:
            表示该动作的 字符串
        """
        return self.env.action_to_human_input(action)


class Gomoku:
    def __init__(self):
        self.board_size = 11
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1  # 黑棋为1，白棋为-1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        return 0 if self.player == 1 else 1  # 先手返回0，后手返回1

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        self.board[x][y] = self.player

        done = self.is_finished()

        reward = 1 if done else 0  # 当前玩家获胜，奖励为1

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = np.where(self.board == 1, 1.0, 0.0)
        board_player2 = np.where(self.board == -1, 1.0, 0.0)
        board_to_play = np.full((11, 11), self.player, dtype="int32")
        return np.array([board_player1, board_player2, board_to_play])  # 3, 11, 11

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))  # 左下、正下、右下、正右
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 如果没有子在这个位置上，不需要考虑这个位置。
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-坐标处的值，i行，j列
                player = self.board[i][j]
                # 检查一行中是否有5个
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                            y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # 如果一行中有5个，则存储所有子的位置，返回值
                        if count == 5:
                            return True
        return not has_legal_actions  # 没有合法位置，游戏结束

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y
