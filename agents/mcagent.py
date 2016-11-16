from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict
from policy import epsilon_greedy_policy, random_policy

class MCAgent(object):

	def __init__(self, env, gamma = 0.9):
		self.env = env
		self.gamma = 0.9
	
	def on_policy_epsilon_greedy(self, episodes = None, epsilon = 0.1, epsilon_decay = False):

		if episodes == None:
			raise ValueError('Enter number of episodes')

		# GLIE Monte-Carlo
		
		# Q-values 
		Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

		# Store count of every state, action pair over all episodes
		returns_count = defaultdict(float)

		# Get policy
		get_action, policy = epsilon_greedy_policy(Q, epsilon, nA = self.env.action_space.n)

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
				
				# Get index of first occurence and sum up discounted returns
				# till end of episode
				first_idx = next(idx for idx, ele in enumerate(episode)
						 if ele[0] == state and ele[1] == action)
				G = sum(x[2] * (self.gamma ** idx) for idx, x in enumerate(episode[first_idx:]))

				# Increase returns count for that pair by one
				returns_count[pair] += 1.0

				# Update Q value for the pair using incremental returns
				Q[state][action] += (1/returns_count[pair]) * (G - Q[state][action])

		return get_action, policy, Q

	def off_policy_importance_sampling(self, episodes = None, epsilon = None, epsilon_decay = False, weighted_sampling = True):

		if epsilon == None and epsilon_decay != False:
			raise ValueError('Set epsilon decay to be True only if using epison greedy behavior policy')

		# Q-values
		Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

		# Cumulative sum of weights given to first n returns
		C = defaultdict(lambda: np.zeros(self.env.action_space.n))

		# Choose behavior policy as epsilon greedy if epsilon is not None else
		# choose it as uniformly random policy (uniform over action space)
		if epsilon != None:
			get_behavior_action, behavior_policy = epsilon_greedy_policy(Q, epsilon, self.env.action_space.n)
		else:
			get_behavior_action, behavior_policy = random_policy(self.env.action_space.n)

		# Target deterministic greedy policy
		get_target_action, target_policy = self.epsilon_greedy_policy(Q, epsilon = 0, nA = self.env.action_space.n)

		for i in range(1, episodes + 1):
			
			# Print out the current number of episode
			print("\rEpisode {}/{}".format(i, episodes), end = "")

			state = self.env.reset()
			done = False

			# Generate episode
			episode = []

			while not done:
				if epsilon_decay:
					action = get_behavior_action(state, i)
				else:
					action = get_behavior_action(state, 1)

				next_state, reward, done, _ = self.env.step(action)
				episode.append((state, action, reward))
				state = next_state

			# Sum of discounted returns 
			G = 0.0

			# Importance sampling ratio
			W = 1.0

			# Iterate through episode in reverse direction
			for i in range(len(episode))[::-1]:

				state, action, reward = episode[i]
				
				# Update reward 
				G = self.gamma * G + reward 

				# Update cumulative weight
				C[state][action] += W

				# Update Q-value using incremental update
				Q[state][action] += W / (C[state][action]) * (G - Q[state][action])

				# Break loop if action returned by target policy is not as that 
				# returned by behavior policy
				if get_target_action != action:
					break

				# Update weight
				sampling_ratio_denom = behavior_policy(state,i)[action] if epsilon_decay else behavior_policy(state,1)[action]	
				W *= 1./sampling_ratio_denom

		return get_target_action, target_policy, Q
