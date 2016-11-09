import numpy as np

#TODO add comments

class DPAgent(object):

	def __init__(self, env, gamma = 0.9):
		self.env = env

		# Discount factor
		self.gamma = gamma

	def value_iteration(self, theta = 0.000001, dummy_states = []):

		# Initialize Value vector
		V = np.zeros(self.env.nS)

		# Assign constant value to dummy states 
		for state,val in dummy_states:
			V[state] = val

		# Function to get action value vector for each action
		# using Bellman equation
		def get_actionvalue(V,s):
			A = np.full(self.env.nA, -float('inf'))

			for a in self.env.P[s]:
				A[a] = 0
				for prob, next_state, reward, _ in self.env.P[s][a]:
					A[a] += prob * (reward + self.gamma * V[next_state])

			return A

		while True:
			delta = 0
			for s in range(self.env.nS):
				if s in [i[0] for i in dummy_states]:
					continue

				# Assign value to state using Bellman optimality equation
				val = np.max(get_actionvalue(V,s))
				delta = max(delta, np.abs(val - V[s]))
				V[s] = val

			# Stop looping if value doesn't change by more than theta
			if delta < theta:
				break

		# Initialize policy vector
		policy = np.zeros((self.env.nS, self.env.nA))

		# Compute deterministic policy using computed values
		for s in range(self.env.nS):
			best_action = np.argmax(get_actionvalue(V,s))
			policy[s][best_action] = 1.0

		return policy, V
