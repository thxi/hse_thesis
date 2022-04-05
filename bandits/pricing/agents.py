from ctypes import ArgumentError
import numpy as np

from bandits.agents import Agent


class UCB1OAgent(Agent):
    def __init__(self, action_to_price, alpha=1):
        self.alpha = alpha
        self.action_to_price = action_to_price
        self.num_arms = len(action_to_price)

        self.reset()

    def _calc_ucb_range(self, j, i):
        # a helper to calculate the ucb in range j..i including j and i
        T_ji = np.sum(self.bandit_to_num_pulls[j : i + 1])
        X_ji = np.sum((self.bandit_to_mean_conversion[j : i + 1] * self.bandit_to_num_pulls[j : i + 1])) / T_ji
        return X_ji + np.sqrt(2 * np.log(self.t) / T_ji)

    def _choose_arm(self):
        if self.explored_bandits < self.num_arms:
            # not finished exploration phase
            k = self.explored_bandits
            return k

        u = []
        for i in range(self.num_arms):
            u_i = float("inf")
            for j in range(0, i + 1):
                ucb_range = self._calc_ucb_range(j, i)
                u_i = min(ucb_range, u_i)
            u.append(u_i)

        ucb_payoffs = self.action_to_price * u
        k = np.argmax(ucb_payoffs)
        # print(f"{self.bandit_to_mean_conversion=}")
        # print(f"{self.bandit_to_num_pulls=}")
        # print(f"{k=} {ucb_payoffs=}")
        # print()
        return k

    def get_action(self, observation):
        action = self._choose_arm()
        return action

    def update_estimates(self, action, observation, reward):
        price = self.action_to_price[action]
        if price == 0:
            raise ArgumentError("price is zero")
        conversion = reward / price

        old_mean_conversion = self.bandit_to_mean_conversion[action]
        old_count = self.bandit_to_num_pulls[action]
        new_mean_conversion = (old_mean_conversion * old_count + conversion) / (old_count + 1)

        self.bandit_to_mean_conversion[action] = new_mean_conversion
        self.bandit_to_num_pulls[action] = old_count + 1

        self.explored_bandits += 1
        self.t += 1

    def reset(self):
        self.bandit_to_mean_conversion = np.zeros(self.num_arms)
        self.bandit_to_num_pulls = np.zeros(self.num_arms)

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round
