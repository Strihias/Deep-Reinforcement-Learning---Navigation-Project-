# Agent Implementation 

In order to implement this project I have used Deep Q-Networks which is a Value Based method. 
This implementation includes 2 training improvements: 
* Experience Replay 
* Fixed Q-Targets 

## Code implementation 

This implementation consists of the following files: 
* Navigation.ypnb: a Jupyter notebook in which we train the agent. In the notebook, we load the UnityEnvironment (provided by Udacity), initialize state, action_size, state_size
and then train the agent setting a limit of 2000 episodes, using the Deep Q Network algorithm. In our case we were able solve the problem in 462 episodes. 

* model.py: where a PyTorch Q-Network class is implemented. This Network, used by the agent, is composed of:
1. input layer
2. two fully connected hidden layers of 128 cells each
3. output layer 

* dqn_agent.py: here we have defined the DQN Agent and the Replay Buffer used by it. The implementation of both follows the methodology described in the paper [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## Algorithm 

1. Starts by initializing the Replay Buffer and setting inital weights for the neural networks
2. For each episode within the max_episodes given it:
  - Resets Environment
  - Gets current state from enviroment For each step in maximum number of timesteps per episode:

  - Picks an action using state using an epslion-greedy algorithm
* Executes this action in the enviroments to obtain rewards, next_state, done

* Stores this experience in the replay buffer If timestep matches EVERY_UPDATE
* Sample random batch of experiences from replay buffer
*  Get predicted Q values from target network using next_states
* Compute target for current states using rewards + (gamma * Q_targets_next * (1 - dones))
* Get expected values from local model using states and actions
* Compute MSE Loss with expected values and target values
* Minimize loss using Adam Optimization, backprop and step
* Perform soft update of local network with target network and TAU value

* Keep looping until the average score over last 100 episodes >= 13
* Save weights of local network to file

## Hyperparameters 

The following parameters were used: 

For the Agent:
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size (Initially 64)
GAMMA = 0.99            # discount factor (Initially 0.99)
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate (Initially 5e-4)
UPDATE_EVERY = 4        # how often to update the network

For the DQN: 
n_episodes = 2000       # maximum number of training episodes
max_t = 1000            # maximum number of timesteps per episode
eps_start = 1.0         # starting value of epsilon, for epsilon-greedy action selection
eps_end = 0.01          # minimum value of epsilon
eps_decay =0.995        # multiplicative factor (per episode) for decreasing epsilon

This tuning led to solving the environment in 462 episodes scoring an avg of 13.04

![alt text](https://github.com/Strihias/Deep-Reinforcement-Learning---Navigation-Project-/blob/main/diagram.jpg "Performance Diagram")


# Future Work
* Better tuing of hyperparameters


