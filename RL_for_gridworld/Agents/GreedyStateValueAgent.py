"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Agents.Agent import Agent
from Gridworld import move

class GreedyStateValueAgent(Agent):
    """Uses State Values to inform its behavior."""

    def choose_action(self, state):
        """Determines the value of each new state by taking all the actions and then chooses an action 
        using an epsilon-greedy policy.
        """ 
        if self.epsilon > np.random.rand():
            return np.random.choice(range(self.num_actions))
        else:
            state_values = [] 
            for action in range(self.num_actions):
                state_values.append(self.get_value(move(action, state)[0]))
            state_values = np.array(state_values)
            indices = np.where(state_values == state_values.max())[0]
            return np.random.choice(indices)

    def display_model(self):
        """Visualizes the state_value function."""
        self.model.display_model()