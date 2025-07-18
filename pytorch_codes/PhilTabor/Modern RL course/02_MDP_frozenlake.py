import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# LEFT=0 DOWN=1 RIGHT=2 UP=3

#SFFF
#FHFH
#FFFH
#HFFG

policy = {0:1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:1, 10:1, 13:2, 14:2}

env = gym.make("FrozenLake-v1")

n_games= 1000
win_pct = []
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()[0]
    score = 0
    while not done:
        action = policy[obs]
        obs, reward, done, info, _ = env.step(action)
        score += reward

    scores.append(score)
    if i%10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()

