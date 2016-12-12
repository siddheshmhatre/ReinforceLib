import numpy as np

def epsilon_greedy_policy(Q, epsilon = 0.1, nA = None):

		# Function that returns the probability distribution over 
		# actions for a particular state
		def policy(state, episode_no = float('inf')):
			probs = np.ones(nA, dtype = float) * ((epsilon / episode_no) / nA)
			best_action = np.argmax(Q[state])
			probs[best_action] += (1.0 - (epsilon / episode_no))
			return probs

		# Function that returns action based on probability distribution
		def get_action(state, episode_no = float('inf')):
			probs = policy(state, episode_no)
			return np.random.choice(np.arange(len(probs)), p = probs)

		return get_action, policy

def random_policy(nA = None):

		def policy(state, episode_no = None):
			return np.ones(nA, dtype = float) / nA
			
		def get_action(state, episode_no = None):
			probs = policy(state)
			return np.random.choice(np.arange(len(probs)), p = probs)

		return get_action, policy
