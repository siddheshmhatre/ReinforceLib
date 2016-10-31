import numpy as np
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld(discrete.DiscreteEnv):

	"""
	GridWorld environment from Sutton and Barto. Implementation mirrors that of https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py
	"""


	def __init__(self, shape = [4,4]):
		
		if not isinstance(shape,(list,tuple)) or not len(shape) == 2:
			raise ValueError('shape must be list or tuple of length 2')

		self.shape = shape
		nS = np.prod(shape)
		nA = 4

		max_y = shape[0]
		max_x = shape[1]

		P = {s : {a : [] for a in range(nA)} for s in range(nS)}
		grid = np.arange(nS).reshape(shape)
		it = np.nditer(grid, flags = ['multi_index'])

		while not it.finished:

			s = it.iterindex
			y, x = it.multi_index

			is_done = lambda x: x == 0 or x == nS - 1 
			reward = 0.0 if is_done(s) else -1.0 

			if is_done(s):
				P[s][UP] = [(1.0, s, reward, True)]
				P[s][RIGHT] = [(1.0, s, reward, True)]
				P[s][DOWN] = [(1.0, s, reward, True)]
				P[s][LEFT] = [(1.0, s, reward, True)]
			else:
				ns_up = s if y == 0 else s - max_x
				ns_right = s if x == (max_x - 1) else s + 1
				ns_down = s if y == (max_y - 1 ) else s + max_x
				ns_left = s if x == 0 else s - 1
				P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
				P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
				P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
				P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

			it.iternext()

		isd = np.ones(nS) / nS

		self.P = P

		super(GridWorld, self).__init__(nS, nA, P, isd)

	def _render(self, mode = 'human', close = False):
		pass
