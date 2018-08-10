"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np

class ReinforceTabular():
    """Class for learning optimal behavior using the policy gradient methods."""

    def __init__(self, alpha = 0.01, gamma = 1, num_actions = 4):

        def value_factory():
            """Used to initialize the values to zeros."""
            return [1/num_actions] * num_actions  

        self.action_probs = defaultdict(value_factory)
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions
    
    def get_value(self, state):
        """Determine the action probabilities for a given state.."""
        return self.action_probs[self.convert_to_immutable(state)]
    
    def convert_to_immutable(self, state):
        """Converts the numpy array into a immutable tuple."""
        reshaped_state = state.reshape(1, state.shape[0]*state.shape[1]*state.shape[2]).ravel()
        return tuple(reshaped_state.tolist())

    def batch_update(self, states, rewards, actions):
        """Update the model using policy gradient Reinforce update method.
            theta = theta + derlog(policy)*reward
        """

        one_hot_actions = np.zeros((len(actions), self.num_actions))
        # Convert action to one hot
        for row, action in enumerate(actions):
            one_hot_actions[row][action] = 1

        # Calculate dicounted rewards
        discounted_rewards = self.discount_and_normalize_rewards(rewards)

        # Update the action_probs by taking the derivative of the log loss function of the softmax probability 
        # distribution and multiplying it by discounted reward (score function)
        for row, state in enumerate(states):
            self.action_probs[self.convert_to_immutable(state)] = self.action_probs[self.convert_to_immutable(state)] + discounted_rewards[row] * (one_hot_actions[row] - self.get_value(state)) 
    

    def discount_and_normalize_rewards(self, episode_rewards):
        """Discount and normalize rewards."""
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    def display_model(self):
        """Display the action with the highest probability in each square. This can be used to trace the path 
        to the final destination.
        """
        actions = ['D', 'U', 'L', 'R']
        best_action_visuals = np.ndarray((4, 4), dtype=object)
        for k, v in self.action_probs.items():
            index = np.where(np.asarray(k).reshape((3, 4, 4))[0] == 1)
            best_action_visuals[index[0][0]][index[1][0]] = actions[int(np.array(v).argmax())]
        
        print(best_action_visuals)