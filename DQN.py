import numpy as np
import torch
import torch.nn.functional as F
from net import Qnet


class DQN:
    """ DQN算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, update_num, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.update_num = update_num  # 目标网络更新间隔
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def choose_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, data):
        states = torch.tensor(data['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(data['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(data['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(data['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(data['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        targets = rewards + self.gamma * max_next_values * (1 - dones)
        self.update_parameters(values, targets)  # 更新参数

    # 更新网路参数
    def update_parameters(self, values, targets):
        mse_loss = F.mse_loss(values, targets)
        loss = torch.mean(mse_loss)  # 均方误差损失函数
        self.optimizer.zero_grad()  # 梯度置0
        loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.update_num == 0:
            self.update_target()  # 更新目标网络参数
        self.count += 1

    # 更新目标网络
    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
