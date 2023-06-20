import json
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
from DQN import DQN
import dqn_config
from buffer import Buffer



class DqnAgent:
    def __init__(self, env_name='MountainCar-v0'):
        """
        :param env_name: 环境名称
        """
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.buffer = Buffer(dqn_config.buffer_size)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.DQN = DQN(state_dim, dqn_config.hidden_dim, action_dim, dqn_config.lr, dqn_config.gamma,
                       dqn_config.epsilon,
                       dqn_config.target_update, self.device)
        self.log = open('log/dqn.txt', 'w', encoding='UTF-8')

    # 开始train
    def fit(self, model_path='model/dqn.pth'):
        self.log.write('开始训练DQN算法模型~\n')
        print('开始训练DQN算法模型~')
        reward_list = []  # 记录每次游戏奖励信息的列表
        i = 1
        success_num = 0  # 成功计数
        # 总共玩iteration_times轮游戏
        while i <= dqn_config.iteration_times:
            # 初始化环境与状态
            state = self.env.reset()[0]  # 状态
            sum_reward = 0  # 这轮游戏的累计奖励
            # 进行一轮游戏
            while True:
                # 选择当前动作
                action = self.DQN.choose_action(state)
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                sum_reward += reward
                # 添加缓存信息
                self.buffer.add(state, action, reward, next_state, terminated)
                # 更新状态
                state = next_state

                # 缓存区达一定大小后再开始更新网路参数哦
                if len(self.buffer) >= dqn_config.batch_size * 2:
                    data = self.buffer.sample(dqn_config.batch_size)
                    self.DQN.update(data)

                # 这轮游戏结束
                if terminated:
                    break
            reward_list.append(sum_reward)
            self.log.write(f'第{i}轮游戏结束，reward={sum_reward}\n')
            if i % 50 == 0:
                mean_reward = np.mean(reward_list[-50:])
                print(f'第{i - 49}~{i}轮游戏的平均奖励为{np.mean(reward_list[-50:]):.3f}')
                if mean_reward >= -110:
                    success_num += 1
                    # 提前终止，保存较好模型
                    if success_num == 3:
                        break
                else:
                    success_num = 0  # 置零
            i += 1
        self.save_model(model_path)
        with open('result/dqn_reward.json', 'w', encoding='UTF-8') as f:
            json.dump(reward_list, f, ensure_ascii=False)
        print('训练完毕，log信息在log文件夹中~')
        print(f'模型已保存到{model_path}中~')

    def save_model(self, model_path):
        torch.save(self.DQN.target_q_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.DQN.target_q_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.DQN.q_net.load_state_dict(torch.load(model_path, map_location=self.device))

    def show_result(self):
        with open('result/dqn_reward.json', 'r', encoding='UTF-8') as f:
            reward_list = json.load(f)
            f.close()
        episodes_list = list(range(len(reward_list)))
        plt.figure(1)
        plt.plot(episodes_list, reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('DQN on {}'.format(self.env_name))
        plt.savefig("result/dqn1.png")

        plt.figure(2)
        plt.plot(episodes_list, reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.ylim(-200, -80)
        plt.title('DQN on {}'.format(self.env_name))
        plt.savefig("result/dqn2.png")
        plt.show()


if __name__ == '__main__':
    agent = DqnAgent()
    agent.fit()
    agent.show_result()
