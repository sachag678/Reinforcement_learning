"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Agents.Agent import Agent
from Gridworld import move

class GreedyStateValueAgent(Agent):

    def choose_action(self, state):
        """Determines the value of each new state by taking all the actions and then chooses an action 
        using an epsilon-greedy policy.
        """ 
        if self.epsilon > np.random.rand():
            return np.random.choice(range(self.num_actions))
        else:
            state_values = [] 
            for action in range(self.num_actions):
                state_values.append(self.get_state_value(move(action, state)[0]))
            state_values = np.array(state_values)
            indices = np.where(state_values == state_values.max())[0]
            return np.random.choice(indices)

    def get_state_value(self, state):
        """Get the state value using the model."""
        return self.model.get_value(state)
    
    def update(self, states, rewards, actions):
        """Update the model using the saved games states, rewards and actions."""
        self.model.batch_update(states, rewards, actions)
    
    def show_state_value(self):
        """Visualizes the state_value function."""
        self.model.show_state_value()