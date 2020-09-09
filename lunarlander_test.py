import gym
import numpy as np
from lunarlander_agent import Lunarlander
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = Lunarlander(state_size = 8, action_size = 4, random_seed = 0)
    # load the weights from file
    # agent.__q_network.load_state_dict(torch.load('ll_checkpoint.pth'))
    agent.load_q_network_state(state_path = 'll_checkpoint.pth')
    RUN_EPISODE = 100
    score = np.zeros(RUN_EPISODE)

    for i in range(RUN_EPISODE):
        state = env.reset()
        run_score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            env.render()

            state, reward, done, _ = env.step(action)
            run_score += reward
        
        print(i, "th run, score = ", run_score)
        score[i] = run_score

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score)), score)
    plt.ylabel('Score')
    plt.xlabel('Step #')
    plt.show()
    print("Average score = ", np.mean(score))

    env.close()