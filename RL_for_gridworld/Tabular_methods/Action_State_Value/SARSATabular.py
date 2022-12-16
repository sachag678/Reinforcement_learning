"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np
from Tabular_methods.Action_State_Value.StateActionValueTabular import StateActionValueTabular

class SARSATabular(StateActionValueTabular):
    """Learns optimal behavior using the SARSA update."""

    def batch_update(self, states, rewards, actions):
        """Implements the SARSA update.
            Q(S, A) = Q(S, A) + alpha ((R + gamma * Q(S', A')) - Q(S, A))
        """
        states = states[:-1]  # drop last state to account for the combining state-action and state-value
        for index, (state, reward, action) in enumerate(zip(states, rewards, actions)):
            if index < len(states) - 1:
                actions_vals = self.get_value(states[index + 1])
                action_val = actions_vals[actions[index + 1]]
                sarsa_target = reward + self.gamma * (action_val)
            else:
                sarsa_target = reward
            
            self.state_action_values[self.convert_to_immutable(state)][action] = self.state_action_values[self.convert_to_immutable(state)][action] + self.alpha * (sarsa_target - self.state_action_values[self.convert_to_immutable(state)][action])