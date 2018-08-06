"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np

class SARSATabular():
    """Learns optimal behavior using the SARSA update."""

    def __init__(self, alpha = 0.01, gamma = 0.95, num_actions = 4):

        def value_factory(value):
            """Used to initialize the values."""
            return lambda: value

        self.state_action_values = defaultdict(value_factory(np.zeros(num_actions)))
        self.alpha = alpha
        self.gamma = gamma
    
    def get_value(self, state):
        self.state_action_values[self.convert_to_immutable(state)]
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2]).ravel()
        return tuple(reshaped_state.tolist())

    def batch_update(self, states, rewards, actions):
        pass
