import torch
import numpy as np
import gym

env = gym.make('CartPole-v0')
from Policy import Policy

policy = Policy()

policy.load_state_dict(torch.load('trained_weights/base1.pt'))

state = env.reset()
for t in range(10000):
    action, _ = policy.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 
