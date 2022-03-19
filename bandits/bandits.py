from abc import ABC, abstractmethod

import numpy as np


class Bandit(ABC):
    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def num_arms(self):
        pass


class BernoulliBandit(Bandit):
    def __init__(self, probs):
        super().__init__()
        self.probs = probs

    def pull(self, k):
        return np.random.binomial(1, self.probs[k])

    def num_arms(self):
        return len(self.probs)
