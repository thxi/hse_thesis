import numpy as np
from gym import Env, spaces

from bandits.bandits import BinomialBanditEnv


class PricingBernoulliBanditEnv(Env):
    def __init__(self, num_arms, dist, p_min=1, p_max=17, n_customers=100):
        super(PricingBernoulliBanditEnv, self).__init__()

        self.num_arms = num_arms
        self.dist = dist  # scipy dist
        self.p_min = p_min
        self.p_max = p_max

        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        self.action_to_price = np.linspace(p_min, p_max, num_arms)
        self.mus = 1 - dist.cdf(self.action_to_price)
        self.b_bandit = BinomialBanditEnv(n=n_customers, probs=self.mus)

        self.max_reward = np.max(self.mus * self.action_to_price)

    def step(self, action):
        assert self.b_bandit.action_space.contains(action)

        observation, conversion_reward, done, info = self.b_bandit.step(action)
        price = self.action_to_price[action]
        reward = conversion_reward * price
        return observation, reward, done, info

    def reset(self):
        return 0
