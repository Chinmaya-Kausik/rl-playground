
import torch

def update_policy_reinforce_static_baseline(policy, batch_returns, batch_log_prob_actions, optimizer, batch_size, baseline=0):
    loss = 0
    
    for run in range(batch_size):
        # Detach to get a copy that won't affect gradients
        batch_returns[run] = batch_returns[run].detach()

        # Calculate the loss using reinforce with static baseline
        n = len(batch_returns[run])
        loss -= torch.sum(torch.mul((batch_returns[run]-baseline), batch_log_prob_actions[run][:n]))/batch_size
    
    # Backprop and take one gradient step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Return loss as float
    return loss.item()

