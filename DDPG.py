from net import PolicyNet, ValueNetwork
import torch
import numpy as np
import torch.nn.functional as F


class DDPG:
    """ DDPG算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, epsilon, actor_lr, critic_lr, update_num, gamma, device):
        # 初始化actor, critic网络与目标网络
        # actor网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化target_actor
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # critic网络
        self.critic = ValueNetwork(state_dim, hidden_dim, action_dim).to(device)
        # 初始化target_critic
        self.target_critic = ValueNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.epsilon = epsilon  # 探索率
        self.update_num = update_num  # 目标网络更新间隔
        self.count = 0  # 计数器,记录更新次数
        self.action_dim = action_dim
        self.device = device

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.epsilon * np.random.randn(self.action_dim)
        mask = action <= -1
        action[mask] = -0.99
        mask = action >= 1
        action[mask] = 0.99
        return action

    def update(self, data):
        states = torch.tensor(data['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(data['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(data['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(data['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(data['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        self.update_critic_parameters(states, actions, targets)
        self.update_actor_parameters(states)
        if self.count % self.update_num == 0:
            self.update_target()
        self.count += 1

    def update_critic_parameters(self, states, actions, targets):
        critic_out = self.critic(states, actions)
        mse_loss = F.mse_loss(critic_out, targets)
        loss = torch.mean(mse_loss)  # 均方误差损失函数
        # 梯度置零
        self.critic_optimizer.zero_grad()
        loss.backward()  # 反向传播更新参数
        self.critic_optimizer.step()

    def update_actor_parameters(self, states):
        actor_out = self.actor(states)
        critic_out = self.critic(states, actor_out)
        loss = -torch.mean(critic_out)  # 均方误差损失函数
        # 梯度置零
        self.actor_optimizer.zero_grad()
        loss.backward()  # 反向传播更新参数
        self.actor_optimizer.step()

    # 更新目标网络
    def update_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
