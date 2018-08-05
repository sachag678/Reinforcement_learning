"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict

class MonteCarloTabular():
    """Learns optimum behavior using the iterative tabular MC method."""

    def __init__(self, alpha=0.1, gamma = 0.99):
        """Initialize."""
        self.state_values = defaultdict(int)
        self.alpha = alpha
        self.gamma = gamma
    
    def get_value(self, state):
        """Return the state value for a given state.
            This requires reshaping the state to be a 1-D array, and then making it immutable by turning 
            it into a list and then a tuple.
        """
        return self.state_values[self.convert_to_immutable(state)]
    
    def batch_update(self, states, rewards, actions):
        """Update the model using the states, rewards."""
        discounted_rewards = self.discount_reward(rewards)
        for state, reward in zip(states, discounted_rewards):
            self.state_values[self.convert_to_immutable(state)] = self.state_values[self.convert_to_immutable(state)] + self.alpha(reward - self.get_value(state))
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2])
        return tuple(reshaped_state.tolist())
    
    def discount_reward(self, rewards):
        """Calculate the future discounted reward."""
        return rewards



