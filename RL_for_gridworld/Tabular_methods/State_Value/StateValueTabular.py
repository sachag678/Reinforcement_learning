"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StateValueTabular():
    """Abstract clas for State Value Tabular methods."""

    def __init__(self, alpha=0.01, gamma = 0.95):
        """Initialize."""

        def value_factory(value):
            """Used to initialize the values."""
            return lambda: value

        self.state_values = defaultdict(value_factory(int()))
        self.alpha = alpha
        self.gamma = gamma
    
    def get_value(self, state):
        """Return the state value for a given state.
            This requires reshaping the state to be a 1-D array, and then making it immutable by turning 
            it into a list and then a tuple.
        """
        return self.state_values[self.convert_to_immutable(state)]
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0] * state.shape[1] * state.shape[2]).ravel()
        return tuple(reshaped_state.tolist())
    
    def display_model(self):
        """Visualizes the state_value function."""
        state_value_visuals = np.zeros((4, 4))
        for k, v in self.state_values.items():
            index = np.where(np.asarray(k).reshape((3, 4, 4))[0] == 1)
            state_value_visuals[index[0][0]][index[1][0]] = v
        
        sns.heatmap(state_value_visuals, linewidth=0.5)
        plt.show()
    
    def batch_update(self, states, rewards, actions):
        """Update the model in the manner specific to the type of updater."""
        raise NotImplementedError('Please implement this function!')

    
    
    
    