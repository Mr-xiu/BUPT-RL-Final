from agent import DQNAgent, DDPGAgent


def train_dqn():
    agent = DQNAgent()
    agent.fit()
    agent.show_result()


def train_ddpg():
    agent = DDPGAgent()
    agent.fit()
    agent.show_result()


if __name__ == '__main__':
    train_dqn()
    train_ddpg()
