import random
import numpy as np


class Buffer:
    """ 经验回放池 """

    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.header_index = 0  # 当前队列头的索引

    # 将数据加入buffer
    def add(self, state, action, reward, next_state, done):
        # 若buffer未满，则追加
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.header_index] = (state, action, reward, next_state, done)
        # 更新索引
        self.header_index = (self.header_index + 1) % self.buffer_size

    def sample(self, batch_size):
        # 从buffer中随机采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        data = {
            'states': np.array(state),
            'actions': action,
            'rewards': reward,
            'next_states': np.array(next_state),
            'dones': done
        }
        return data

    def __len__(self):
        return len(self.buffer)
