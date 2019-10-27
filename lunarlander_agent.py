import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from collections import namedtuple, deque

class TorchQNetwork(nn.Module):
    ''' Q-Network class, defines topology of the NN in Torch, implement forward method'''

    def __init__(self, state_size, action_size, random_seed, layer1_units = 64, layer2_units = 64):
        '''Initialize 3 layers of linear NN '''
        super(TorchQNetwork, self).__init__()
        self.__seed = torch.manual_seed(random_seed)
        self.__layer1 = nn.Linear(state_size, layer1_units)
        self.__layer2 = nn.Linear(layer1_units, layer2_units)
        self.__layer3 = nn.Linear(layer2_units, action_size)

    def forward(self, state):
        '''Implement forward method to provide action value based on state input'''
        x = func.relu(self.__layer1(state))
        x = func.relu(self.__layer2(x))
        return self.__layer3(x)

class Memory():
    ''' Memory to store Lunarlander agent experiences'''
    def __init__(self, action_size, buffer_size, batch_size, random_seed):
        self.__action_size = action_size
        self.__batch_size = batch_size
        self.__seed = random.seed(random_seed)
        self.__memory = deque(maxlen = buffer_size)
        self.__experience = namedtuple("experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.__device = "cpu"

    def add(self, state, action, reward, next_state, done):
        single_expr = self.__experience(state, action, reward, next_state, done)
        self.__memory.append(single_expr)

    def random_sample(self):
        ''' Random generate a sample from memory'''
        exprs = random.sample(self.__memory, k = self.__batch_size)

        states = torch.from_numpy(np.vstack([expr.state for expr in exprs if expr is not None])).float().to(self.__device)
        actions = torch.from_numpy(np.vstack([expr.action for expr in exprs if expr is not None])).long().to(self.__device)
        rewards = torch.from_numpy(np.vstack([expr.reward for expr in exprs if expr is not None])).float().to(self.__device)
        next_states = torch.from_numpy(np.vstack([expr.next_state for expr in exprs if expr is not None])).float().to(self.__device)
        dones = torch.from_numpy(np.vstack([expr.done for expr in exprs if expr is not None])).float().to(self.__device)

        return (states, actions, rewards, next_states, dones)

    def length(self):
        return len(self.__memory)
 
class Lunarlander():
    '''Lunarlander agent, include methods of step, act, Q-learn, update'''
    def __init__(self, state_size, action_size, random_seed):
        self.__state_size = state_size
        self.__action_size = action_size
        self.__random_seed = random_seed
        self.__seed = random.seed(random_seed)
        self.__device = "cpu"
        self.__current_step = 0

        self.set_agent_details()

        # Create Q network, current __q_network and next __next_q_network
        self.__create_q_network()

        # Set up memory for replay
        self.__memory = Memory(self.__action_size, self.__memory_buffer_size, self.__batch_size, self.__random_seed)


    def set_agent_details(self, gamma = 0.99, tau = 1e-3, learning_rate = 5e-4, update_frequency = 4, memory_buffer_size = 100000, batch_size = 64):
        self.__gamma = gamma                            # Discount factor
        self.__tau = tau                                # Soft update target parameter
        self.__learning_rate = learning_rate            # Learning rate
        self.__update_frequency = update_frequency      # Every nextwork after every N batch
        self.__memory_buffer_size = memory_buffer_size  # Memory replay buffer size
        self.__batch_size = batch_size                  # Mini batch size

    # Methods to create Q network
    def __create_q_network(self):
        self.__q_network = TorchQNetwork(self.__state_size, self.__action_size, self.__random_seed).to(self.__device)
        self.__next_q_network = TorchQNetwork(self.__state_size, self.__action_size, self.__random_seed).to(self.__device)
        self.__optimizer = optim.Adam(self.__q_network.parameters(), lr = self.__learning_rate)

    # action
    def act(self, state, eps = 0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)
        self.__q_network.eval()
        with torch.no_grad():
            action_values = self.__q_network(state)
        self.__q_network.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.__action_size))

    # step
    def step(self, state, action, reward, next_state, done):
        self.__memory.add(state, action, reward, next_state, done)
        self.__current_step += 1
        if self.__current_step % self.__update_frequency == 0:
            #check if time to update
            if self.__memory.length() > self.__batch_size:
                exprs = self.__memory.random_sample()
                # print("current_step:", self.__current_step, ", memory:", self.__memory.length(), ", batch_size:", self.__batch_size, ", expr:", exprs)
                self.__learn(exprs)
    #learn
    def __learn(self, exprs):
        states, actions, rewards, next_states, dones = exprs
        # Get maximum predicted q values
        q_next_targets = self.__next_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute current q_targets with vectorization
        q_targets = rewards + (self.__gamma * q_next_targets * ( 1- dones))

        # from q_network estimated q value
        q_est = self.__q_network(states).gather(1, actions)

        loss = func.mse_loss(q_est, q_targets)

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        # Update q network model parameters
        self.__update_network()

    def __update_network(self):
        '''Update __q_network and __next_q_network '''
        for next_param, current_param in zip(self.__next_q_network.parameters(), self.__q_network.parameters()):
            next_param.data.copy_(self.__tau * current_param.data + (1.0 - self.__tau) * next_param.data)

    def load_q_network_state(self, state_path):
        self.__q_network.load_state_dict(torch.load(state_path))

    def save_q_network_state(self, state_path): 
        torch.save(self.__q_network.state_dict(), state_path)