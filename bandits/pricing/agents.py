import numpy as np

from bandits.agents import Agent
from bandits.pricing.bandits import PricingBernoulliBandit


class UCB1OAgent(Agent):
    def __init__(self, bandit: PricingBernoulliBandit, alpha=1):
        self.bandit = bandit
        self.alpha = alpha

        self.reset()

    def _calc_ucb_range(self, j, i):
        # a helper to calculate the ucb in range j..i including j and i
        T_ji = np.sum(self.bandit_to_num_pulls[j : i + 1])
        X_ji = np.sum((self.bandit_to_mean_conversion[j : i + 1] * self.bandit_to_num_pulls[j : i + 1])) / T_ji
        return X_ji + np.sqrt(2 * np.log(self.t) / T_ji)

    def _choose_bandit(self):
        if self.explored_bandits < self.bandit.num_arms():
            # not finished exploration phase
            k = self.explored_bandits
            return k

        u = []
        for i in range(self.bandit.num_arms()):
            u_i = float("inf")
            for j in range(0, i + 1):
                ucb_range = self._calc_ucb_range(j, i)
                u_i = min(ucb_range, u_i)
            u.append(u_i)

        ucb_payoffs = self.bandit.arms * u
        k = np.argmax(ucb_payoffs)
        # print(f"{self.bandit_to_mean_conversion=}")
        # print(f"{self.bandit_to_num_pulls=}")
        # print(f"{k=} {ucb_payoffs=}")
        # print()
        # print()
        return k

    def take_action(self):
        k = self._choose_bandit()
        reward = self.bandit.pull(k)
        price = self.bandit.arms[k]
        if price == 0:
            raise NotImplementedError("price is zero")
        conversion = reward / price
        self._update_rewards(k=k, conversion=conversion, reward=reward)
        self.explored_bandits += 1
        self.t += 1
        return reward

    def _update_rewards(self, k, conversion, reward):
        old_mean_conversion = self.bandit_to_mean_conversion[k]
        old_count = self.bandit_to_num_pulls[k]
        new_mean_conversion = (old_mean_conversion * old_count + conversion) / (old_count + 1)

        self.bandit_to_mean_conversion[k] = new_mean_conversion
        self.bandit_to_num_pulls[k] = old_count + 1
        self.bandit_to_conversions[k].append(conversion)
        self.reward_history.append(reward)

    def reset(self):
        self.reward_history = []
        self.bandit_to_mean_conversion = np.zeros(self.bandit.num_arms())
        self.bandit_to_num_pulls = np.zeros(self.bandit.num_arms())
        self.bandit_to_conversions = [[] * self.bandit.num_arms() for _ in range(self.bandit.num_arms())]

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round
