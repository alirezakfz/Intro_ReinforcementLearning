import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from replay_memory import ReplayBuffer

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_inout_dims = self.calcultae_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_inout_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calcultae_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv1(conv1))
        conv3 = F.relu(self.conv1(conv2))
        # Conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], 1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        return actions
    
    def save_checkpoint(self):
        print('.... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                  replace=1000, algo=None, env_name=None, 
                   chkpt_dir="temp/dqn" ):
        self.gamma = gamma
        self.lr = lr
        self.epsilon= epsilon
        self.n_actions =n_actions
        self.input_dims = input_dims
        self.batch_size=batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = DeepQNetwork(lr, n_actions, input_dims=input_dims,
                                   name=env_name+'_'+algo+'_q_eval',
                                   chkpt_dir=chkpt_dir)
        
        self.q_next = DeepQNetwork(lr, n_actions, input_dims=input_dims,
                                   name=env_name+'_'+algo+'_q_next',
                                   chkpt_dir=chkpt_dir)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]   # dims -> batch_size * n_actions
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


from utils import make_env, plot_learning_curve

if __name__ == "__main__":
    env = make_env("PongNoFrameskip-v4")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    print("ENV action space: ", env.action_space)
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions= env.action_space.n,
                     mem_size=10000,
                     eps_min = 0.1,
                     batch_size=32,
                     replace=1000,
                     eps_dec=1e-5,
                     chkpt_dir="models/",
                     algo = "DQNAgent",
                     env_name="PongNoFrameskip-v4"
                      )
    
    if load_checkpoint:
        agent.load_models()
    
    fname = agent.algo + '_' + agent.env_name + '_lr'+str(agent.lr) + '_' +\
    '_'+str(n_games)+ 'games'
    figure_file = 'plots/' + fname + '.png'
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score:', score, 
              'average score %.1f best score %.1f epsilon %.2f'%(avg_score, best_score, agent.epsilon),
              'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        eps_history.append(agent.epsilon)
        plot_learning_curve(steps_array, scores, eps_history, figure_file)


