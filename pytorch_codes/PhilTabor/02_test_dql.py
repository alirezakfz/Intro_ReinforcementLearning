import gymnasium as gym
from deep_q_learning import Agent
import numpy as np
from plot_learning_curve import plot_learning_curve

if __name__ == '__main__':
    render = False
    env = env = gym.make('LunarLander-v2', render_mode='human' if render else None)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.003)
    
    scores, eps_history = [], []
    n_games = 1000

    print("start_")
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward=reward, state_=observation_, done=done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('epsilon', i, 
              'score %0.2f'%score, 
              'average score %0.2f'%avg_score,
              'epsilon %.2f'%agent.epsilon)
    
    x = [i+1 for i in range(n_games)]
    file_name = 'lunar_lander_2020.png'
    plot_learning_curve(x, scores, eps_history, file_name)


