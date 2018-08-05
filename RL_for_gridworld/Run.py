"""Sacha Gunaratne | github@sachag678"""
import numpy as np
from Gridworld import init_state, move
from RandomAgent import RandomAgent

def run(max_episodes, agent):
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
                print('Game over')
            
            state = new_state
    
    return agent

if __name__ == '__main__':
    r_agent = RandomAgent()
    r_agent = run(2, r_agent)



