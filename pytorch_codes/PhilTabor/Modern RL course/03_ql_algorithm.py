import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


class Agent():
    def __init__(self, lr, gamma, n_actions, 
                 n_states, eps_start, eps_end, eps_dec) -> None:
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states= n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}
        self.init_Q()

    
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[state, a] for a in range(self.n_actions)])
            action = np.argmax(actions)
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state_,a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)
        self.Q[(state, action)] += self.lr*(reward + self.gamma*self.Q[(state, a_max)] - self.Q[(state, action)])
        self.decrement_epsilon()

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = Agent(lr=0.001, gamma=0.9, 
                  eps_start=1.0, eps_end=0.01, 
                  eps_dec=0.999995, 
                  n_actions=env.action_space.n, 
                  n_states=env.observation_space.n)
    
    scores =[]
    win_pct = []
    n_games = 50000


    for i in range(n_games):
        done = False
        obs = env.reset()[0]
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info, _ = env.step(action)
            agent.learn(obs, action, reward, obs_)
            score += reward
            obs = obs_
        
        scores.append(score)
        if i%100 ==0:
            win_avg = np.mean(scores[-100:])
            win_pct.append(win_avg)
            if i%1000 == 0:
                print("episode ", i, "win pct %.2f"%win_avg, "epsilon %.2f"% agent.epsilon)

    print("plotting")
    plt.plot(win_pct)
    plt.show()


