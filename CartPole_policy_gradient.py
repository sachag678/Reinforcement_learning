"""Learn how to play cartpole with policy gradient and keras."""

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as back

import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

# Environement states
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


def discount_and_normalize_rewards(episode_rewards):
    """Discount and normalize rewards."""
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


def create_model(input_size, output_size, hidden_dims=[10]):
    """Create Model."""
    model = Sequential()
    model.add(Dense(hidden_dims[0], input_shape=(input_size, ), activation='relu', kernel_initializer='glorot_normal'))
    # self.model.add(Dropout(0.2))
    for h_dim in hidden_dims[1:]:
        model.add(Dense(h_dim, activation='relu', kernel_initializer='glorot_normal'))

    model.add(Dense(2, activation='relu', kernel_initializer='glorot_normal'))
    # self.model.add(Dropout(0.2))

    model.add(Dense(output_size, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def create_train_function(alpha, model, output_dim):
    """Create the training function."""
    action_prob = model.output

    action_one_hot_placeholder = back.placeholder(shape=(None, output_dim),
                                                  name="action_one_hot")

    discounted_reward_placeholder = back.placeholder(shape=(None, ),
                                                     name='discount_reward')

    log_prob = back.sum(action_one_hot_placeholder * back.log(action_prob), axis=1)

    loss = back.mean(- log_prob * discounted_reward_placeholder)

    adam = keras.optimizers.Adam(lr=alpha)

    updates = adam.get_updates(params=model.trainable_weights,
                               loss=loss)

    return back.function(inputs=[model.input, action_one_hot_placeholder, discounted_reward_placeholder],
                         outputs=[],
                         updates=updates)

# Hyperparams
max_episodes = 1000
alpha = 0.01
gamma = 0.95

# initialize
allRewards, episode_states, episode_actions_one_hot, episode_rewards = [], [], [], []

model = create_model(state_size, action_size)
train_fn = create_train_function(alpha, model, action_size)

for episode in range(max_episodes):

    state = env.reset()

    # env.render()

    while True:
        action_prob = model.predict(state.reshape([1, 4]), batch_size=1)

        action = np.random.choice(range(action_size), p=action_prob.ravel())

        new_state, reward, done, info = env.step(action)

        episode_states.append(state)

        action_ = np.zeros(action_size)
        action_[action] = 1

        episode_actions_one_hot.append(action_)
        episode_rewards.append(reward)

        if done:
            # Calculate sum reward
            episode_rewards_sum = np.sum(episode_rewards)

            allRewards.append(episode_rewards_sum)

            total_rewards = np.sum(allRewards)

            # Mean reward
            mean_reward = np.divide(total_rewards, episode + 1)

            maximumRewardRecorded = np.amax(allRewards)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", maximumRewardRecorded)

            discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

            episode_states = np.vstack(np.array(episode_states))
            episode_actions_one_hot = np.vstack(np.array(episode_actions_one_hot))

            train_fn([episode_states, episode_actions_one_hot, discounted_episode_rewards])

            # Reset the transition stores
            episode_states, episode_actions_one_hot, episode_rewards = [], [], []

            break

        state = new_state
