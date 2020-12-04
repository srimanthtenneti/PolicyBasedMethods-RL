"""                                                                      Reinforce Implmentation - Policy Based method                                                                     """
# Importing the necessary libraries
#__________________________________

import numpy as np # For array operations
import matplotlib.pyplot as plt # Visulization tool

# Functions from the Pytorch API
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#OpenAI gym for the environment
import gym

#Utility
from collections import deque

#Policy
from Policy import Policy 

#_________________________________


# Creating an environment instance
#_________________________________

env = gym.make('CartPole-v0') #Instance of the Cart-Pole-v0 environment
env.seed(3) # The environments seed parameter

#____________________________________

# Choosing the compute platform
#______________________________

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Viewing the environment information
#____________________________________
print("\r\n")
print("Cart-Pole env V0")
print("_________________")
print("State shape  : " , env.observation_space.shape) #State dimensions 
print("Action space : " , env.action_space.n) #Action space dimensions

      
#_________________________________________

# Viewing the model
#_________________________________________

policy = Policy().to(device)
print("\r\n")
print("Policy model")
print("_____________")
print(policy)
print("\r\n")
#_________________________________________

# Training the model
#_________________________________________
optimizer = optim.Adam(policy.parameters() , lr = 1e-2)

def reinforce(n_episodes=2000,max_T=1000,gamma=0.99,print_every=100):
    
    scores_deque = deque(maxlen=100)
    scores = []
    print("Training ...\r\n")
    
    for i_episode in range(1 , n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_T):
            action,log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state,reward,done,_ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a*b for a,b in zip(discounts , rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {} \t Avg Score {:.2f} \t'.format(i_episode , np.mean(scores_deque)))
            torch.save(policy.state_dict() , "trained_weights/base2.pt")
            
        if np.mean(scores_deque) >= 195.0:
            print("Environment solved !!!! :)")
            print("Training complete")
            break
            
    return scores
 
#Uncomment the below line to train the model   
scores = reinforce()

#_________________________________________
