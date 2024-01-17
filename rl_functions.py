"""
Setting up the base RL Algorithm class
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
     
class RLAlg(object):
    
    def __init__(self, normalize_ret, max_eps_length, discount_factor, train_batch_size,
                 test_batch_size, policy, optimizer):
        super().__init__()
        
        self.normalize_ret = normalize_ret
        self.discount_factor = discount_factor
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_eps_length = max_eps_length
        self.policy = policy
        self.optimizer = optimizer
        
        
    def calculate_returns(self, rewards):

        # Initialize return vector r and running return value R
        returns = []
        R = 0

        # Discount rewards and add up
        for r in reversed(rewards):
            R = r + R * self.discount_factor
            returns.append(R)

        # Cast to a torch tensor
        returns = torch.tensor(returns[::-1])

        if self.normalize_ret:
            returns = (returns - returns.mean())/returns.std()

        return returns

    # Function to train a policy
    def train(self, env, device):

        # Standard training initialization
        self.policy.train()
        batch_returns = []
        batch_log_prob_actions = torch.zeros(self.train_batch_size, self.max_eps_length)
        mean_reward = 0

        for eps in range(self.train_batch_size):

            # Initialize various vectors
            rewards = []
            terminated = False
            time = 0

            state, info = env.reset()

            while (not terminated) and (time < self.max_eps_length):

                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(device)
                except:
                    print(state)

                action_pred = self.policy(state)

                action_prob = F.softmax(action_pred, dim = -1)

                dist = distributions.Categorical(action_prob)

                action = dist.sample()

                batch_log_prob_actions[eps][time] = dist.log_prob(action)

                state, reward, terminated, truncated, info = env.step(action.item())

                rewards.append(reward)

                mean_reward += reward/float(self.train_batch_size)

                time += 1

            returns = self.calculate_returns(rewards)
            batch_returns.append(returns)

        loss = self.update_policy(batch_returns, batch_log_prob_actions)

        return loss, mean_reward

    # Function to evaluate policies
    def evaluate(self, env, device):

        # Declare eval mode
        self.policy.eval()
        mean_reward = 0

        for eps in range(self.test_batch_size):

            terminated = False
            episode_reward = 0
            time = 0

            state, info = env.reset()

            while (not terminated) and (time < self.max_eps_length):

                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(device)
                except:
                    print(state)

                with torch.no_grad():

                    action_pred = self.policy(state)

                    action_prob = F.softmax(action_pred, dim = -1)

                action = torch.argmax(action_prob, dim = -1)

                state, reward, terminated, truncated, info = env.step(action.item())

                episode_reward += reward

                time += 1

            mean_reward += episode_reward/float(self.test_batch_size)

        return mean_reward
    
    def update_policy(self, batch_returns, batch_log_prob_actions):
        raise NotImplementedError

    
     