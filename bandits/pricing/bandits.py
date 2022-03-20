import numpy as np

from bandits.bandits import Bandit, BernoulliBandit


class PricingBernoulliBandit(Bandit):
    def __init__(self, K, dist, p_min=1, p_max=17):
        super().__init__()
        self.K = K
        self.dist = dist  # scipy dist
        self.p_min = p_min
        self.p_max = p_max

        self.arms = np.linspace(p_min, p_max, K + 1)[1:]
        self.mus = 1 - dist.cdf(self.arms)
        self.b_bandit = BernoulliBandit(probs=self.mus)

    def pull(self, k):
        conversion = self.b_bandit.pull(k)
        reward = conversion * self.arms[k]
        return reward

    def num_arms(self):
        return self.K
