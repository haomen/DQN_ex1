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

### Gym Setup
Box2D installation: `pip install gym[Box2D]`

### How to run:
python run_to_test.py

## Components
- lunarlander_agent.py: defines agent, Deep Q-Network Learning model
- lunarlander_train.py: run this code to generate converged model
- lunarlander_test.py:  run this code to picked trained model, and verify results, of course, provide perfect landing.
