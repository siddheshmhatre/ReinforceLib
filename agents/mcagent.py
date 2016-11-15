from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict

class MCAgent(object):

	def __init__(self, env, gamma = 0.9):
		self.env = env
		self.gamma = 0.9

	def epsilon_greedy_policy(self, Q, epsilon = 0.1, nA = None):

		# Function that returns the probability distribution over 
		# actions for a particular state
		def policy(state, episode_no = float('inf')):
			probs = np.ones(nA) * ((epsilon / episode_no) / nA)
			best_action = np.argmax(Q[state])
			probs[best_action] += (1.0 - (epsilon / episode_no))
			return probs

		# Function that returns action based on probability distribution
		def get_action(state, episode_no = float('inf')):
			probs = policy(state, episode_no)
			return np.random.choice(np.arange(len(probs)), p = probs)

		return get_action, policy

	def on_policy_epsilon_greedy(self, episodes = None, epsilon = 0.1, epsilon_decay = False):

		if episodes == None:
			raise ValueError('Enter number of episodes')

		# GLIE Monte-Carlo
		
		# Q-values 
		Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

		# Store count of every state, action pair over all episodes
		returns_count = defaultdict(float)

		# Get policy
		get_action, policy = self.epsilon_greedy_policy(Q, epsilon, self.env.action_space.n)

		for i in range(1, episodes + 1):
			
			# Print out the current number of episode
			print("\rEpisode {}/{}".format(i, episodes), end = "")

			state = self.env.reset()
			done = False

			# Generate episode
			episode = []

			while not done:
				if epsilon_decay:
					action = get_action(state, i)
				else:
					action = get_action(state, 1)

				next_state, reward, done, _ = self.env.step(action)
				episode.append((state, action, reward))
				state = next_state

			state_action = set((tuple(ele[0]),ele[1]) for ele in episode)

			for state, action in state_action:
				pair = (state,action)
				
				# Get index of first occurence and sum up return
				# till end of episode
				first_idx = next(idx for idx, ele in enumerate(episode)
						 if ele[0] == state and ele[1] == action)
				G = sum(x[2] for x in episode[first_idx:])

				# Increase returns count for that pair by one
				returns_count[pair] += 1.0

				# Update Q value for the pair using incremental returns
				Q[state][action] += (1/returns_count[pair]) * (G - Q[state][action])

		return get_action, policy, Q
