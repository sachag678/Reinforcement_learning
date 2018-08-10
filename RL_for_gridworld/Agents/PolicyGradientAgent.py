"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Agents.Agent import Agent
from Gridworld import move

class PolicyGradientAgent(Agent):
    """Uses Policy Gradient methods to inform its behavior."""

    def choose_action(self, state):
        """Determines the value of each new state by taking all the actions and then chooses an action 
        using an epsilon-greedy policy.
        """ 
        action_probs = self.get_value(state)
        return np.random.choice(range(len(action_probs)), p = action_probs)

    def display_model(self):
        """Visualizes the state_value function."""
        self.model.display_model()