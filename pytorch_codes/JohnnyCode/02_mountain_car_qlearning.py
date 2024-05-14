import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):

    env = gym.make('MountainCar-v0',  render_mode='human' if render else None)

    state = env.reset()[0]  # Starting position
    terminated = False      # True when fall in hole or reached goal
    
    rewards = 0
    
    while(not terminated and rewards > -1000):
        action = env.action_space.sample()
        
        new_state,reward,terminated,_,_ = env.step(action)
        
        state = new_state
        
    env.close()
    
if __name__ == '__main__':
    run(1, is_training=False, render=True)