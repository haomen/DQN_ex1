# Lunar Lander

From OpenAI box2d project [LunarLander](https://gym.openai.com/envs/LunarLander-v2/)
----

## Objective
To understand Deep Q-Network Learning.

## Results
This result gives average score of 200+ per 100 runs, able to complete OpenAI LunarLander mission.

## Env Requirements
- Python3.6
- Pytorch
- Matplotlib
- OpenAI Gym
- numpy
- pandas (for Plotting.ipynb only)

### Gym Setup
Box2D installation: `pip install gym[Box2D]`

### How to run:
python run_to_test.py

## Components
- lunarlander_agent.py: defines agent, Deep Q-Network Learning model
- lunarlander_train.py: run this code to generate converged model
- lunarlander_test.py:  run this code to picked trained model, and verify results, of course, provide perfect landing.

To run the whole experiment conducted for the write-up, a few jupyter notebook files are included. Note that these part of codes are set up to run on Google Colab. To run the code under a new user account with Google, please follow the comments in the code to set up path and mount folders.

- lunarlander_main.ipynb: the main model training and testing for the best hyperparameter selected
- lunarlander_main_set1_gamma.ipynb: trials for different gamma values
- lunarlander_main_set1_network.ipynb: trials for different NN structure setups
- Plotting.ipynb: helper scripts to plot all results presented in write-up.