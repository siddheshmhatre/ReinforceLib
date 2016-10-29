import numpy as np

#TODO add comments

class DPAgent(object):

	def __init__(self, env, gamma = 0.9):
		self.env = env
		self.gamma = gamma

	def value_iteration(self, theta = 0.000001):

		V = np.zeros(env.nS)

		def get_actionvalue(V,s):
			A = np.zeros(env.nA)

			for a in range(env.nA):
				for prob, next_state, reward, _ in env.P[s][a]:
					A[a] += prob * (reward + self.gamma * V[next_state])

			return A

		while True:
			delta = 0

			for s in range(env.nS):
				val = np.max(get_actionvalue(V,s))
				delta = np.max(delta, np.abs(val - V[s]))
				V[s] = val

			if delta < theta:
				break

		policy = np.zeros((env.nS, env.nA))

		for s in range(env.nS):
			best_action = np.argmax(get_actionvalue(V,s))
			policy[s][best_action] = 1.0

		return policy, V

	def policy_iteration():
		pass

