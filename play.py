import gymnasium as gym
from agent import DQNAgent, DDPGAgent


def play_dqn(model_path='model/dqn.pth', env_name='MountainCar-v0'):
    agent = DQNAgent()
    agent.load_model(model_path=model_path)
    env = gym.make(env_name, render_mode="human")
    print('-' * 50 + '\n开始游戏（DQN算法）~')
    state = env.reset()[0]
    step = 0  # 走的步数
    sum_reward = 0
    while True:
        env.render()
        action = agent.DQN.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        step += 1
        state = next_state
        if terminated:
            env.render()
            print('已到达终点！！！')
            break
        elif truncated:
            print('游戏失败~~~')
            break
    print(f'共走了{step}步，获得的总奖励为{sum_reward}~')


def play_ddpg(model_path='model/', env_name='MountainCarContinuous-v0'):
    agent = DDPGAgent()
    agent.load_model(model_path=model_path)
    env = gym.make(env_name, render_mode="human")
    print('-' * 50 + '\n开始游戏（DDPG算法）~')
    state = env.reset()[0]
    step = 0  # 走的步数
    sum_reward = 0
    while True:
        env.render()
        action = agent.DDPG.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        sum_reward += reward
        step += 1
        state = next_state
        if terminated:
            env.render()
            print('已到达终点！！！')
            break
        elif truncated:
            print('游戏失败~~~')
            break
    print(f'共走了{step}步，获得的总奖励为{sum_reward}~')


if __name__ == '__main__':
    play_dqn()
    play_ddpg()
