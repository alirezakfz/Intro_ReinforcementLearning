import matplotlib.pyplot as plt
from agent import Agent
from agent_Qlearning import Agent as QAgent
from agent_ExpectedSarsa import Agent as ESAgent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

# avg_list=list(avg_rewards)
# points=np.arange(len(avg_list))
# plt.plot(points, avg_list)



agent = QAgent()
avg_rewards, best_avg_reward = interact(env, agent)

# avg_list=list(avg_rewards)
# points=np.arange(len(avg_list))
# plt.plot(points, avg_list)


agent = ESAgent()
avg_rewards, best_avg_reward = interact(env, agent)