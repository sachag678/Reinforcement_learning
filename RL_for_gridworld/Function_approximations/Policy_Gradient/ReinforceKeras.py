"""Sacha Gunaratne | github@sachag678"""
from collections import defaultdict
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as back

class ReinforceKeras():
    """Class for learning optimal behavior using the policy gradient methods."""

    def __init__(self, input_size, alpha = 0.01, gamma = 1, num_actions = 4):


        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = num_actions

        self.create_model(input_size, num_actions)
        self.create_train_function()
    
    def create_train_function(self):
        """Create the training function."""
        action_prob = self.model.output

        action_one_hot_placeholder = back.placeholder(shape=(None, self.num_actions),
                                                      name="action_one_hot")

        discounted_reward_placeholder = back.placeholder(shape=(None, ),
                                                         name='discount_reward')

        log_prob = back.sum(action_one_hot_placeholder * back.log(action_prob), axis=1)

        loss = back.mean(- log_prob * discounted_reward_placeholder)

        adam = keras.optimizers.Adam(lr=self.alpha)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fcn = back.function(inputs=[self.model.input, action_one_hot_placeholder, discounted_reward_placeholder],
                             outputs=[],
                             updates=updates)

    def create_model(self, input_size, output_size, hidden_dims=[10]):
        """Create Model with relu activation layers in between and a softmax activation on the final layer.
        Used Xavier initialization for the layers.

        Default model structure is input_size-->10-->4-->output_size.
        """
        self.model = Sequential()
        self.model.add(Dense(hidden_dims[0], input_shape=(input_size, ), activation='relu', kernel_initializer='glorot_normal'))

        for h_dim in hidden_dims[1:]:
            self.model.add(Dense(h_dim, activation='relu', kernel_initializer='glorot_normal'))

        self.model.add(Dense(4, activation='relu', kernel_initializer='glorot_normal'))

        self.model.add(Dense(output_size, activation='softmax', kernel_initializer='glorot_normal'))

    def get_value(self, state):
        """Determine the action probabilities for a given state.."""
        return self.model.predict(state.reshape(1, state[0] * state[1] * state[2]), batch_size=1)
    
    def batch_update(self, states, rewards, actions):
        """Update the model using policy gradient Reinforce update method.
            theta = theta + derlog(policy)*reward
        """
        one_hot_actions = np.zeros((len(actions), self.num_actions))
        # Convert action to one hot.
        for row, action in enumerate(actions):
            one_hot_actions[row][action] = 1

        # Calculate dicounted rewards.
        discounted_rewards = self.discount_and_normalize_rewards(rewards)

        # Convert the episodes and actions into numpy arrays.
        states = np.vstack(np.array(states))
        one_hot_actions = np.vstack(one_hot_actions)

        # Update the model using the in-built train function.
        self.train_fcn([states, one_hot_actions, discounted_rewards])

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
        pass