import numpy as np
from gym.envs.toy_text import discrete

class GamblersProblem(discrete.DescreteEnv):

	"""
	GamblersProblem is an environment from Sutton and Barto
	"""


	def __init__(self, p_h = 0.4, target = 100):

		if not isinstance(target, int):
			raise ValueError('Target should be an integer')

		if not ( p_h >= 0 and p_h <= 1):
			raise ValueError('Enter valid value for probability of heads')

		self.p_h = 0.4

		nS = target + 1
		nA = target / 2  + 1
		P = {}
		reward = lambda s: 1.0 if s == target else 0.0
		is_done = lambda s: if s == 0 or s == target

		for s in range(target+1):
			
			if s == target or s == 0:
				P = {a : [(1.0, s, reward(s), True)] for a in range(nA)}
			else:
				P = {a : [(1.0, s, reward(s), False)] for a in range(nA)}

				for a in range(1, min(s, target - s) + 1):
					P[s][a] = [(p_h, s + a, reward(s), is_done(s + a)), (1 - p_h, s - a, reward(s), is_done(s - a))] 

		self.P = P
		
		isd = np.ones(nS) / (target - 1) 
		isd[0] = 0.0
		isd[-1] = 0.0

		super(GamblersProblem, self).__init__(nS, nA, P, isd)
		
	def _render(self, mode = 'human', close = False):
		pass
