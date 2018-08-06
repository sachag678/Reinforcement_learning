"""Sacha Gunaratne | github@sachag678"""

class Agent():
    """Abstract implementation of an Agent for GridWorld."""
    
    def __init__(self, num_actions = 4, model = None, epsilon = None):
        """Initialize."""
        self.num_actions = num_actions
        self.model = model
        self.epsilon = epsilon

    def choose_action(self, state):
        """Choose an action given a state."""
        raise NotImplementedError('This should be implemented.')
    
    def update(self, states, rewards, actions):
        """Update the behavior mechanism."""
        pass
    
    def get_state_action_values(self, state):
        """Get the value for each action given the state."""