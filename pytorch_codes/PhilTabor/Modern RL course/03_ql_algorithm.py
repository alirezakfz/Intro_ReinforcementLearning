import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class QL:
    def __init__(self, n_actions, n_states, alfa, gamma, ep_max, ep_min) -> None:
        self.alfa = alfa
        self.gamma = gamma
        self.ep_max = ep_max
        self.ep_min = ep_min
        self.q = np.zeros((n_states, n_actions))
    
    def update(self, state, action, new_state, reward):
        max_a = np.argmax(self.q[new_state,:])
        self.q[state, action] = self.q[state, action] + self.alfa*(reward + self.gamma*self.q[new_state,max_a] - self.q[state, action] )
    
    def best_action(self, state):
        index = np.argmax(self.q[state,:])
        return index + 1
    
    def dec_epsilon(self):
        if(self.ep_max > self.ep_min):
            self.ep_max = self.ep_max - self.ep_max*0.001
        else:
            self.ep_max = self.ep_min
    

class Agent:
    env = gym.make("FrozenLake-v1")
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    n_games = 10000
    alfa = 0.001
    gamma = 0.9
    ep_max = 1.0
    ep_min = 0.01

    win_pct = []
    scores = []
    ql = QL(n_actions=n_actions, 
            n_states=n_states, 
            gamma=gamma, 
            alfa=alfa, 
            ep_max=ep_max,
            ep_min=ep_min
            )
    


    for i in range(n_games):
        done = False
        obs = env.reset()[0]
        score = 0
        while not done:
            if(np.random.random() > ep_max):
                action = ql.best_action(obs)
            else:
                action = np.random.choice(range(env.action_space.n))

            new_obs, reward, done, info, _ = env.step(action)
            ql.update(
                action=action,
                state=obs,
                new_state=new_obs,
                reward=reward
                )
            
            score += reward

        ql.dec_epsilon()
        scores.append(score)
        if i%10 == 0:
            average = np.mean(scores[-10:])
            win_pct.append(average)



agent = Agent()

plt.plot(agent.win_pct)
plt.show()

