"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np
from Tabular_methods.Action_State_Value.StateActionValueTabular import StateActionValueTabular

class QLearningTabular(StateActionValueTabular):
    """Learns optimal behavior using the Q Learning update."""

    def batch_update(self, states, rewards, actions):
        """Implements the Q Learning update.
            Q(S, A) = Q(S, A) + alpha ((R + gamma * max(Q(S', A'))) - Q(S, A))
        """
        for index, (state, reward, action) in enumerate(zip(states, rewards, actions)):
            if index < len(states) - 1:
                actions_vals = np.array(self.get_value(states[index + 1]))
                max_action_val = np.random.choice(actions_vals[np.where(actions_vals == actions_vals.max())[0]])
                sarsa_target = reward + self.gamma * (max_action_val)
            else:
                sarsa_target = reward
            
            self.state_action_values[self.convert_to_immutable(state)][action] = self.state_action_values[self.convert_to_immutable(state)][action] + self.alpha * (sarsa_target - self.state_action_values[self.convert_to_immutable(state)][action])