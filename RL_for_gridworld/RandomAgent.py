"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Agent import Agent

class RandomAgent(Agent):
    """An Agent that behaves randomly and doesn't learn."""

    def choose_action(self, state):
        return np.random.choice(range(self.num_actions))

