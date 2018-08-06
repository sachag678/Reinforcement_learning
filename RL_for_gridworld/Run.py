"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Gridworld import init_state, move, oh_no_dragon, found_gold
from RandomAgent import RandomAgent
from GreedyStateValueAgent import GreedyStateValueAgent
from MonteCarloTabular import MonteCarloTabular

def train(max_episodes, agent):
    """Run episodes of gridworld and train the agent.

        params:
                max_episodes - the maximum number of episodes
                agent - the agent that will be acting in the gridworld
        
        return:
                agent - a trained agent
    """

    for _ in range(max_episodes):
        states, rewards, actions = [], [], []
        state, done = init_state()

        while not done:

            new_state, reward, done = move(agent.choose_action(state),state)

            states.append(new_state)
            rewards.append(reward)
            actions.append(actions)

            if done:
                agent.update(states, rewards, actions)
            
            state = new_state
    
    return agent

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

if __name__ == '__main__':
    model = MonteCarloTabular()
    agent = GreedyStateValueAgent(model = model, epsilon=0.2)
    for i in range(10):
        agent = train(100, agent)
        agent.epsilon = 0
        test(agent)
        agent.epsilon = 0.2
    
    agent.show_state_value()
    

    



