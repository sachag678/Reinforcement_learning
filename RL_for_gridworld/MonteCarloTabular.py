"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np

class MonteCarloTabular():
    """Learns optimum behavior using the iterative tabular MC method."""

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
    
    def batch_update(self, states, rewards, actions):
        """Update the model using the states, rewards using the iterative MC update.
            V(s_t) = V(s_t) + alpha(G_t - V(s_t))
        """
        discounted_rewards = self.discount_reward(rewards)
        for state, reward in zip(states, discounted_rewards):
            self.state_values[self.convert_to_immutable(state)] = self.state_values[self.convert_to_immutable(state)] + self.alpha * (reward - self.get_value(state))
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2]).ravel()
        return tuple(reshaped_state.tolist())
    
    def discount_reward(self, rewards):
        """Calculate the future discounted reward."""
        discounted_rewards = []
        for _ in range(len(rewards)):
            temp_r = 0
            for index, r in enumerate(rewards):
                temp_r += (self.gamma**index)*r
            discounted_rewards.append(temp_r)
        return discounted_rewards
    
    def show_state_value(self):
        """Visualizes the state_value function."""
        state_value_visuals = np.zeros((4,4))
        for k, v in self.state_values.items():
            index = np.where(np.asarray(k).reshape((3,4,4))[0]==1)
            state_value_visuals[index[0][0]][index[1][0]] = v
        
        print(state_value_visuals)


