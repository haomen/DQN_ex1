import random
import numpy as np
import torch
from model import QNetwork

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
        self.__q_network = QNetwork(self.__state_size, self.__action_size, self.__random_seed)

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