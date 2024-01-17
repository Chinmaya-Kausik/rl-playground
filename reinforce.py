from rl_functions import *
import torch

class Reinforce(RLAlg):
    
    def __init__(self, normalize_ret, max_eps_length, discount_factor, train_batch_size,
                 test_batch_size, policy, optimizer):
        super().__init__(normalize_ret, max_eps_length, discount_factor, train_batch_size,
                 test_batch_size, policy, optimizer)

    def update_policy(self, batch_returns, batch_log_prob_actions, baseline=0):
        
        loss = 0

        for run in range(self.train_batch_size):
            # Detach to get a copy that won't affect gradients
            batch_returns[run] = batch_returns[run].detach()

            # Calculate the loss using reinforce with static baseline
            n = len(batch_returns[run])
            loss -= torch.sum(torch.mul((batch_returns[run]-baseline), batch_log_prob_actions[run][:n]))/self.train_batch_size

        # Backprop and take one gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return loss as float
        return loss.item()

