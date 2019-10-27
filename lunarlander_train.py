import gym
import numpy as np
import matplotlib.pyplot as plt 
import torch
from collections import deque
from lunarlander_agent import Lunarlander

def trival_landing(env, agent, render = True, plot = True):
    '''
    Try 10 times landing with 200 steps of move in each land try, render in gym and display
    '''
    TRY_SIZE = 10
    scores = np.zeros(TRY_SIZE)
    for i in range(TRY_SIZE):
        state = env.reset()
        score = 0

        for j in range(200):
            action = agent.act(state)
            if render:
                env.render()
            state, reward, done, _ = env.step(action)
            score += reward

            if done:
                break
        scores[i] = score

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Try #')
        plt.show()


def train_model(env, agent, n_episodes = 1000, max_t = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995, render = True, plot = True):
    # agent.set_agent_details(gamma = 0.99, tau = 1e-3, learning_rate = 5e-4, update_frequency = 4, memory_buffer_size = 100000, batch_size = 64):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen = 100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay * eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#         if np.mean(scores_window)>=200.0:
#             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
    print("saving trained results..")
    agent.save_q_network_state('ll_checkpoint.pth')
    # torch.save(agent.qnetwork_local.state_dict(), 'll_checkpoint.pth')
#             break
    return scores

if __name__ == "__main__":
    # Initialize environment and Lunarlander agent
    env = gym.make('LunarLander-v2')
    env.seed(0)
    env_state_size = env.observation_space.shape[0]
    env_action_size = env.action_space.n

    agent = Lunarlander(state_size = env_state_size, action_size = env_action_size, random_seed = 0)
    
    # # random landing for 10 times, basic understanding of gym env
    # trival_landing(env, agent, render = False, plot = False)

    # train
    train_model(env, agent)

    env.close()