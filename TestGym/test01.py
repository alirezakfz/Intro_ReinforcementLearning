import gym
env = gym.make('CartPole-v0')#to test different models change "CartPole-V0" to different models
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()