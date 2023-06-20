import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    """ Q网络类 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
