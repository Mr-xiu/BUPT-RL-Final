import random
import numpy as np


class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, buffer_max_size):
        self.buffer = []
        self.buffer_max_size = buffer_max_size
        self.now_index = 0  # 当前队列头的索引

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):
        # 若buffer未满，则追加
        if len(self.buffer) < self.buffer_max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.now_index] = (state, action, reward, next_state, done)
        # 更新索引
        self.now_index = (self.now_index + 1) % self.buffer_max_size

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
