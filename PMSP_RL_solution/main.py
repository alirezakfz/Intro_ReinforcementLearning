import matplotlib.pyplot as plt
from agent import Agent
from env_pmsp import PMSP1


# from agent_Qlearning import Agent as QAgent
# from agent_ExpectedSarsa import Agent as ESAgent


from monitor import interact


# import gym

import numpy as np

env = PMSP1(8,3,4)
env.reset()
nA=len(env.actions_list)

agent = Agent(nA)

avg_rewards, best_avg_reward = interact(env, agent)

avg_list=list(avg_rewards)
points=np.arange(len(avg_list))
plt.plot(points, avg_list)



#agent = QAgent()
#avg_rewards, best_avg_reward = interact(env, agent)

# avg_list=list(avg_rewards)
# points=np.arange(len(avg_list))
# plt.plot(points, avg_list)


#agent = ESAgent()
#avg_rewards, best_avg_reward = interact(env, agent)