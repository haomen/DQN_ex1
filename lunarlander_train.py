import gym
import numpy as np
import matplotlib.pyplot as plt 
from lunarlander_agent import Lunarlander

def trival_landing(env, agent, render = True):
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Try #')
    plt.show()


if __name__ == "__main__":
    # Initialize environment and Lunarlander agent
    env = gym.make('LunarLander-v2')
    env.seed(0)
    env_state_size = env.observation_space.shape[0]
    env_action_size = env.action_space.n

    agent = Lunarlander(state_size = env_state_size, action_size = env_action_size, random_seed = 0)
    
    # random landing for 10 times, basic understanding of gym env
    trival_landing(env, agent, render = False)

    env.close()