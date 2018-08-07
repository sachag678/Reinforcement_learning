"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Agents.Agent import Agent
from Gridworld import move

class GreedyStateActionValueAgent(Agent):

    def choose_action(self, state):
        """Determines the value of taking each action from a given state and then chooses an action 
        using an epsilon-greedy policy.
        """ 
        if self.epsilon > np.random.rand():
            return np.random.choice(range(self.num_actions))
        else:
            state_action_values = self.get_value(state)
            state_action_values = np.array(state_action_values)
            indices = np.where(state_action_values == state_action_values.max())[0]
            return np.random.choice(indices)

    def get_value(self, state):
        """Get the state value using the model."""
        return self.model.get_value(state)
    
    def update(self, states, rewards, actions):
        """Update the model using the saved games states, rewards and actions."""
        self.model.batch_update(states, rewards, actions)
    
    def display_model(self):
        """Visualizes the state_value function."""
        self.model.display_model()