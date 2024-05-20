import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v1')

n_games = 1000
win_pct = []  # Keep track of the win percentages
scores =  []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample()
        #print(env.step(action))
        obs, reward, done, info, _ = env.step(action)
        score += reward
    
    scores.append(score)

    if i%10==0:
        average = np.mean(scores[-10:])
        win_pct.append(average)


plt.plot(win_pct)
plt.show()



