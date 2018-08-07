"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np

class SARSATabular():
    """Learns optimal behavior using the SARSA update."""

    def __init__(self, alpha = 0.01, gamma = 1, num_actions = 4):

        def value_factory():
            """Used to initialize the values to zeros."""
            return [0] * num_actions  

        self.state_action_values = defaultdict(value_factory)
        self.alpha = alpha
        self.gamma = gamma
    
    def get_value(self, state):
        """Determine the state action value."""
        return self.state_action_values[self.convert_to_immutable(state)]
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2]).ravel()
        return tuple(reshaped_state.tolist())

    def batch_update(self, states, rewards, actions):
        """Implements the SARSA update.
            Q(S, A) = Q(S, A) + alpha ((R + gamma * Q(S', A')) - Q(S, A))
        """
        for index, (state, reward, action) in enumerate(zip(states, rewards, actions)):
            if index < len(states) - 1:
                actions_vals = self.get_value(states[index + 1])
                action_val = actions_vals[actions[index + 1]]
                sarsa_target = reward + self.gamma * (action_val)
            else:
                sarsa_target = reward
            
            self.state_action_values[self.convert_to_immutable(state)][action] = self.state_action_values[self.convert_to_immutable(state)][action] + self.alpha * (sarsa_target - self.state_action_values[self.convert_to_immutable(state)][action])

    def display_model(self):
        """Display the action with the highest probability in each square. This can be used to trace the path 
        to the final destination.
        """
        actions = ['D', 'U', 'L', 'R']
        best_action_visuals = np.ndarray((4, 4), dtype=object)
        for k, v in self.state_action_values.items():
            index = np.where(np.asarray(k).reshape((3, 4, 4))[0] == 1)
            best_action_visuals[index[0][0]][index[1][0]] = actions[int(np.array(v).argmax())]
        
        print(best_action_visuals)