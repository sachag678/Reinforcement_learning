"""Sacha Gunaratne | github@sachag678"""
import numpy as np
import matplotlib.pyplot as plt

from Gridworld import init_state, move, oh_no_dragon, found_gold
from Agents.RandomAgent import RandomAgent
from Agents.GreedyStateValueAgent import GreedyStateValueAgent
from Agents.GreedyStateActionValueAgent import GreedyStateActionValueAgent
from Tabular_methods.State_Value.MonteCarloTabular import MonteCarloTabular
from Tabular_methods.State_Value.TemporalDifferenceTabular import TemporalDifferenceTabular
from Tabular_methods.Action_State_Value.SARSATabular import SARSATabular

def train(max_episodes, agent):
    """Run episodes of gridworld and train the agent.

        params:
                max_episodes - the maximum number of episodes
                agent - the agent that will be acting in the gridworld
        
        return:
                agent - a trained agent
    """
    episode_rewards = []

    for epoch in range(max_episodes):
        states, rewards, actions = [], [], []
        state, done = init_state()

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done = move(action, state)

            states.append(new_state)
            rewards.append(reward)
            actions.append(action)

            if done:
                episode_rewards.append(np.array(rewards).sum())
                agent.update(states, rewards, actions)
            
            if agent.epsilon >= 0.05:
                agent.epsilon -= 1/max_episodes
            state = new_state
    
    return agent, episode_rewards

def test(agent):
    """Test the trained agent in a single episode of the gridworld game."""

    state, done = init_state()
    num_moves = 0

    while not done:
        new_state, _, done = move(agent.choose_action(state), state)
        num_moves += 1

        if done:
            print('Number of Moves: ', num_moves)
            if oh_no_dragon(new_state):
                print('You lost!')
            else:
                print('Found gold!')
        
        state = new_state

        if num_moves > 20:
            print('Not enough learning!')
            break
    
def moving_average(dist, window):
    """Calculate the moving average of a distribution of numbers given a time window."""
    if window >= len(dist):
        return dist
    ma = []
    for i in range(len(dist) - window):
        ma.append(np.array(dist[i: i + window]).mean())
    return ma

if __name__ == '__main__':
    model = TemporalDifferenceTabular()
    agent = GreedyStateValueAgent(model = model, epsilon=0.5)
    agent, episode_rewards = train(100, agent)
    agent.epsilon = 0
    test(agent)
    
    agent.display_model()

    plt.plot(moving_average(episode_rewards, 10))
    plt.show()





    



