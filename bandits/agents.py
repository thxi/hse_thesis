from abc import ABC, abstractmethod

import numpy as np

from bandits.bandits import Bandit, BernoulliBandit

# some code is taken from
# https://towardsdatascience.com/multi-armed-bandits-upper-confidence-bound-algorithms-with-python-code-a977728f0e2d


class Agent(ABC):
    @abstractmethod
    def take_action(self):
        pass

    def _update_rewards(self, k, reward):
        old_mean_reward = self.bandit_to_mean_rewards[k]
        old_count = self.bandit_to_num_pulls[k]
        new_mean_reward = (old_mean_reward * old_count + reward) / (old_count + 1)

        self.bandit_to_mean_rewards[k] = new_mean_reward
        self.bandit_to_num_pulls[k] = old_count + 1
        self.bandit_to_rewards[k].append(reward)
        self.reward_history.append(reward)


class EpsilonGreedyAgent(Agent):
    def __init__(self, eps, bandit: Bandit):
        # the probability of selecting a random arm
        self.eps = eps
        self.bandit = bandit
        self.reset()

    def _choose_bandit(self):
        should_explore = np.random.binomial(1, self.eps)
        if should_explore == 1:
            # select random
            k = np.random.randint(0, self.bandit.num_arms())
        else:
            # select best
            k = np.argmax(self.bandit_to_mean_rewards)
        return k

    def take_action(self):
        k = self._choose_bandit()
        reward = self.bandit.pull(k)
        self._update_rewards(k=k, reward=reward)
        return reward

    def reset(self):
        self.reward_history = []
        self.bandit_to_mean_rewards = np.zeros(self.bandit.num_arms())
        self.bandit_to_num_pulls = np.zeros(self.bandit.num_arms())
        self.bandit_to_rewards = [[] * self.bandit.num_arms() for _ in range(self.bandit.num_arms())]


class UCB1Agent(Agent):
    def __init__(self, bandit: Bandit, alpha=1, reward_normalization=1):
        self.bandit = bandit
        self.alpha = alpha
        self.reward_normalization = reward_normalization

        self.reset()

    def _choose_bandit(self):
        if self.explored_bandits < self.bandit.num_arms():
            # not finished exploration phase
            k = self.explored_bandits
            return k

        # selecting based on upper confidence bound
        means = self.bandit_to_mean_rewards
        upper = np.sqrt(2 * np.log(self.t + 1) / (self.bandit_to_num_pulls) + 1)
        k = np.argmax(means + self.alpha * upper)
        return k

    def take_action(self):
        k = self._choose_bandit()
        reward = self.bandit.pull(k) / self.reward_normalization
        self._update_rewards(k=k, reward=reward)
        self.explored_bandits += 1
        self.t += 1
        return reward

    def reset(self):
        self.reward_history = []
        self.bandit_to_mean_rewards = np.zeros(self.bandit.num_arms())
        self.bandit_to_num_pulls = np.zeros(self.bandit.num_arms())
        self.bandit_to_rewards = [[] * self.bandit.num_arms() for _ in range(self.bandit.num_arms())]

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round


class ThompsonSamplingBetaAgent(Agent):
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit

        self.reset()

    def _choose_bandit(self):
        alpha = self.bandit_to_mean_rewards * self.bandit_to_num_pulls
        beta = self.bandit_to_num_pulls - alpha
        theta = np.random.beta(alpha + 1, beta + 1)  # sample from beta dist for each arm
        k = np.argmax(theta)
        return k

    def take_action(self):
        k = self._choose_bandit()
        reward = self.bandit.pull(k)
        self._update_rewards(k=k, reward=reward)
        return reward

    def reset(self):
        self.reward_history = []
        self.bandit_to_mean_rewards = np.zeros(self.bandit.num_arms())
        self.bandit_to_num_pulls = np.zeros(self.bandit.num_arms())
        self.bandit_to_rewards = [[] * self.bandit.num_arms() for _ in range(self.bandit.num_arms())]
