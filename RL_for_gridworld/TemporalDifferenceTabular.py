"""Sacha Gunaratne | github@sachag678"""
from StateValueTabular import StateValueTabular

class TemporalDifferenceTabular(StateValueTabular):
    """Learns optimum behavior using the iterative tabular MC method."""
    
    def batch_update(self, states, rewards, actions):
        """Update the model using the states, rewards using the simple TD(0) update.
            V(s_t) = V(s_t) + alpha((R_t+1+gamma*V(s_t+1)) - V(s_t))
        """
        for index, (state, reward) in enumerate(zip(states, rewards)):
            if index < len(states) - 1:
                TD_target = reward + self.gamma * self.get_value(states[index + 1])
            else:
                TD_target= reward
            self.state_values[self.convert_to_immutable(state)] = self.state_values[self.convert_to_immutable(state)] + self.alpha * (TD_target - self.get_value(state))
    