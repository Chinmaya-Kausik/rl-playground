"""
Functions independent of the RL algorithm
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
     

def calculate_returns(rewards, discount_factor, normalize = True):
    
    # Initialize return vector r and running return value R
    returns = []
    R = 0
    
    # Discount rewards and add up
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.append(R)
    
    # Cast to a torch tensor
    returns = torch.tensor(returns[::-1])
    
    if normalize:
        returns = (returns - returns.mean())/returns.std()
        
    return returns

# Function to train a policy
def train(env, policy, optimizer, policy_update_alg, normalize_returns, max_eps_length, discount_factor, batch_size, device):
    
    # Standard training initialization
    policy.train()
    batch_returns = []
    batch_log_prob_actions = torch.zeros(batch_size, max_eps_length)
    mean_reward = 0
    
    for eps in range(batch_size):
    
        # Initialize various vectors
        rewards = []
        terminated = False
        time = 0

        state, info = env.reset()

        while (not terminated) and (time < max_eps_length):

            try:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            except:
                print(state)

            action_pred = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

            dist = distributions.Categorical(action_prob)

            action = dist.sample()

            batch_log_prob_actions[eps][time] = dist.log_prob(action)

            state, reward, terminated, truncated, info = env.step(action.item())
            
            rewards.append(reward)
            
            mean_reward += reward/float(batch_size)
            
            time += 1
       
        returns = calculate_returns(rewards, discount_factor, normalize_returns)
        batch_returns.append(returns)

    loss = policy_update_alg(policy, batch_returns, batch_log_prob_actions, optimizer, batch_size)

    return loss, mean_reward

# Function to evaluate policies
def evaluate(env, policy, max_eps_length, batch_size, device):
    
    # Declare eval mode
    policy.eval()
    mean_reward = 0
    
    for eps in range(batch_size):
        
        terminated = False
        episode_reward = 0
        time = 0

        state, info = env.reset()

        while (not terminated) and (time < max_eps_length):

            try:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
            except:
                print(state)

            with torch.no_grad():

                action_pred = policy(state)

                action_prob = F.softmax(action_pred, dim = -1)

            action = torch.argmax(action_prob, dim = -1)

            state, reward, terminated, truncated, info = env.step(action.item())

            episode_reward += reward

            time += 1
    
        mean_reward += episode_reward/float(batch_size)

    return mean_reward
     