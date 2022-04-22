import numpy as np
from gym import Env, spaces


class BernoulliBanditEnv(Env):
    def __init__(self, probs):
        super(BernoulliBanditEnv, self).__init__()

        self.probs = probs

        self.action_space = spaces.Discrete(len(probs))
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        self.max_reward = np.max(self.probs)

    def step(self, action):
        assert self.action_space.contains(action)

        observation = 0
        reward = np.random.binomial(1, self.probs[action])
        done = False
        info = None
        return observation, reward, done, info

    def reset(self):
        return 0


class BinomialBanditEnv(Env):
    def __init__(self, n, probs):
        super(BinomialBanditEnv, self).__init__()

        self.n = n
        self.probs = probs

        self.action_space = spaces.Discrete(len(probs))
        self.observation_space = spaces.Discrete(1)  # no observations, only rewards

        self.max_reward = np.max(self.probs)

    def step(self, action):
        assert self.action_space.contains(action)

        observation = 0
        reward = np.random.binomial(self.n, self.probs[action]) / self.n
        done = False
        info = None
        return observation, reward, done, info

    def reset(self):
        return 0


# TODO: maybe register environment
# https://sjcaldwell.github.io/2020/05/21/openai-gym-intro.html
