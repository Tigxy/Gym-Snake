import numpy as np

class Discrete():
    def __init__(self, n_actions, seed=None):
        self.dtype = np.int32
        self.n = n_actions
        self.actions = np.arange(self.n, dtype=self.dtype)
        self.shape = self.actions.shape
        self._rng = np.random.default_rng(seed)

    def set_rng(self, rng):
        self._rng = rng

    def contains(self, argument):
        for action in self.actions:
            if action == argument:
                return True
        return False

    def sample(self):
        return self._rng.choice(self.n)
