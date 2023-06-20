import torch
import torch.nn.functional as F


# DQN算法的Q网络
class Qnet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc3.weight.data.normal_(0, 0.3)

    # 前向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc3.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# 价值网络
class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        self.fc3.weight.data.normal_(0, 0.3)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
