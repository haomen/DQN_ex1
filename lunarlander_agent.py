import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

class TorchQNetwork(nn.Module):
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
 
class Lunarlander():
    '''Lunarlander agent, include methods of step, act, Q-learn, update'''
    def __init__(self, state_size, action_size, random_seed):
        self.__state_size = state_size
        self.__action_size = action_size
        self.__random_seed = random_seed
        self.__seed = random.seed(random_seed)

        self.__device = "cpu"

        # Create Q network
        self.create_q_network()

        # Set up memory for replay



    # Methods to create Q network
    def create_q_network(self):
        self.__q_network = TorchQNetwork(self.__state_size, self.__action_size, self.__random_seed)

    # Methods create training method

    # Observation/Perceive

    # Train Q netwrok

    # egreed action

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
            return random.choise(np.arange(self.action_size))

    # weight variable

    # bias variable


    #def create_training_method(self):

# class MemoryBuffer():
#     '''Memorybuffer for training'''


# class 