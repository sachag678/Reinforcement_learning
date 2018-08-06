"""Sacha Gunaratne | github@sachag678"""
import numpy as np
import copy 

# =====================================================
# The following functions define how the game is played.
# =====================================================

PLAYER_DEPTH = 0
GOLD_DEPTH = 1
DRAGON_DEPTH = 2

def init_state(rows = 4, cols = 4, depth = 3):
    """Initialize starting state.

        params:
                rows -  the number of rows on the board
                cols -  the number of columns in the board
                depth  -  the number of levels in the board. i.e., 
                how many other elements exist including the player such as gold, and the dragon.  
        return:
                initial_state -  the initial state of the game
    """
    initial_state = np.zeros((depth, rows, cols))
    # place the player at row = 3, col = 0, depth = 0
    initial_state[PLAYER_DEPTH][rows - 1][0] = 1
    # place the gold at row = 0, col = 3, :depth = 1
    initial_state[GOLD_DEPTH][0][cols -1] = 1
    # place the dragon who will eat you at row = 1, col = 2, depth = 3
    initial_state[DRAGON_DEPTH][1][2] = 1
    return initial_state, False

def move(action, state):
    """Move the player based on the action and return a new state.

        params:
                action - the actions chosen by the player
                state - the current state of the board
        return:
                new_state - the new state after the action has been taken
                reward - the reward for moving into the new state
                done - the boolean indicating whether the game is over 
    """
    # Create new_state
    new_state = copy.deepcopy(state)
    
    # Get the current location of player.
    player_index = np.where(new_state[PLAYER_DEPTH]==1)
    p_row = player_index[0][0]
    p_col = player_index[1][0]

    # Get the Max rows and cols of state.
    max_row = new_state[PLAYER_DEPTH].shape[0]
    max_col = new_state[PLAYER_DEPTH].shape[1]

    if action == 0:  # DOWN 
        if p_row + 1 < max_row:
            new_state[PLAYER_DEPTH][p_row + 1][p_col] = 1
            new_state[PLAYER_DEPTH][p_row][p_col] = 0
    if action == 1:  # UP
        if p_row - 1 >= 0:
            new_state[PLAYER_DEPTH][p_row - 1][p_col] = 1
            new_state[PLAYER_DEPTH][p_row][p_col] = 0
    if action == 2:  # LEFT
        if p_col - 1 >= 0:
            new_state[PLAYER_DEPTH][p_row][p_col - 1] = 1
            new_state[PLAYER_DEPTH][p_row][p_col] = 0
    if action == 3:  # RIGHT
        if p_col + 1 < max_col:
            new_state[PLAYER_DEPTH][p_row][p_col + 1] = 1
            new_state[PLAYER_DEPTH][p_row][p_col] = 0
    
    return new_state, get_reward(new_state), is_terminal(new_state)

def get_reward(state):
    """Calculate reward given a state."""
    if found_gold(state):
        return 100
    elif oh_no_dragon(state):
        return -10
    else:
        return -1.0  
    
def is_terminal(state):
    """Determine if the game is over."""
    return found_gold(state) or oh_no_dragon(state)

def found_gold(state):
    """Checks if the player found the gold."""
    return np.where(state[PLAYER_DEPTH]==1) == np.where(state[GOLD_DEPTH]==1)

def oh_no_dragon(state):
    """Checks if the player was eaten by the dragon."""
    return np.where(state[PLAYER_DEPTH]==1) == np.where(state[DRAGON_DEPTH]==1)
    
if __name__ == '__main__':
    state, done = init_state()
    new_state, reward, done = move(1, state)
    new_state, reward, done = move(3, new_state)
    new_state, reward, done = move(3, new_state)
    new_state, reward, done = move(1, new_state)
    print(done, reward)
    
    

