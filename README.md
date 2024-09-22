# Reinforcement Learning Playground

Playground for various RL experiments. Implementation details so far:
1. **Base class:** There is a base class called RLALg that implements functions shared by all RL Algorithms (calculating the return vector from a sequence of rewards, a training function, an evaluation function, an unimplemented container function for the policy update step).
2. **Vanilla PG with fixed baselines:** Implemented, but you have to pass the policy and optimizer during initialization.

Relevant papers for design choices: [Dropout](https://arxiv.org/abs/2202.11818), [Hyperparameters](https://arxiv.org/pdf/2306.01324.pdf), [Actor Critic methods](https://openreview.net/pdf?id=nIAxjsniDzg), [on-policy rl](https://arxiv.org/pdf/2006.05990.pdf).
