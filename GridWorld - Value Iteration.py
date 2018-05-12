import numpy as np

size = 3
board = np.zeros((size,size))

def update(size, transition_reward, board, i,j):
	'''Updates a specific location by looking at 4 directions and following the bellman optimality equation
	params: size - the size of the board
			transition_reward - the reward for moving to a new state
			board - the current board
			i - location on the board
			j - location on the board
	return:
			the max value of the updated value function using bellmans optimality
	'''
	if(i==0 and j==0):
		return 0
	if(i==0 and j!=size-1):
		val1 = board[i][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==0 and j==size-1):
		val1 = board[i][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j!=size-1 and j!=0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j+1]+transition_reward
		val4 = board[i][j-1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j==0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j+1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i==size-1 and j==size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j==size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j-1]+transition_reward
		val4 = board[i][j]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j==0):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	if(i!=0 and i!=size-1 and j!=0 and j!=size-1):
		val1 = board[i-1][j]+transition_reward
		val2 = board[i+1][j]+transition_reward
		val3 = board[i][j]+transition_reward
		val4 = board[i][j+1]+transition_reward
		return max(val1,val2,val3,val4)
	return 1

def evaluate(transition_reward,size,board):
	'''Updates a board once
	params: size - the size of the board
			transition_reward - the reward for moving to a new state
			board - the current board 
	return: 
			a fully updated board
	'''
	new_board = np.zeros((size,size))
	for i in range(size):
		for j in range(size):
			new_board[i][j] = update(size,transition_reward,board,i,j)
	return new_board

def run(transition_reward,size, cycles):
	'''Runs the update for a set number of cycles and returns the board either when the cycles are over or it 
	has converged. 
	params: size - the size of the board
			transition_reward - the reward for moving to a new state
			cycles - the number of iterations to update
	return:
			the optimal policy
	'''
	board = np.zeros((size,size))
	for i in range(cycles):
		updated_board = evaluate(transition_reward,size,board)
		if abs(updated_board[size-1][size-1]-board[size-1][size-1])<0.0001:
			print('Converged at cycle num: ', i)
			return updated_board
		board = updated_board

	return board

print(run(-1,4,10))
