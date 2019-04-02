# Reinforcement Learning

This repo contains the implementations of the RL algorithms in GridWorld and a variety of openAI domains.

## Folders

The folder structures are broken down as follows:

### Initial_experiments_and_misc
This contains code that is used when experimenting with new ideas, implementing state of the art and is not considered clean code.
However, it is kept as a reminder of the process. 

The folder contains:
  - Cartpole policy gradient
  - Gridworld policy iteration
  - Black jack policy evaluation
  - A simple NN
  - Tic Tac Toe using DQN

### RL_for_gridworld
This folder contains the tabular implementations of:
  - Monte Carlo
  - TD 
  - SARSA
  - Q-Learning
  
and NN (keras) implementations of:
  - Policy Gradient Reinforce (Vanilla) 

which are used to learn the optimum path to the gold while avoiding the monster in gridworld. 

The next steps are to implement the NN versions of the State Value and State Action Value methods using keras.

### Neural Network Implementation
Contains implementations of a NN in C++, python, Fortran to determine how fast things are comparatively. I also used vanilla C++ vs Eigen which is used as the backend of tensorflow. 

