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
        """Update the model using the saved games states, rewards and actions."""
        self.model.batch_update(states, rewards, actions)
    
    def get_value(self, state):
        """Get the state value using the model."""
        return self.model.get_value(state)