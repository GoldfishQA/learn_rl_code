import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible

# global variable
N_STATES = 6 #length
ACTIONS = ['left','right']
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
	table = pd.DataFrame(
		np.zeros((n_states, len(actions))),
		columns = actions,
		)
	return table

build_q_table(N_STATES, ACTIONS)

def choose_action(state, q_table):
	state_actions = q_table.iloc[state, :]
	if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax()
	return action_name

def get_env_feedback(S, A):
	if A == 'right':
		if S ==N_STATES - 2:
			S = 'terminal'
			R = 1
		else:
			S = S + 1
			R = 0
	else:
		R = 0
		if S == 0:
			S_ = S
		else:
			S_ = S -1
	return S_, R
