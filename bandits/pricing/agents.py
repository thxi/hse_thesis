from ctypes import ArgumentError

import numpy as np

from bandits.agents import Agent
from bandits.aggregating_algorithms import AggregatingAlgorithm
from bandits.online_models import SimpleLinearRegressor


class UCB1OAgent(Agent):
    def __init__(self, action_to_price, alpha=1):
        self.alpha = alpha
        self.action_to_price = action_to_price
        self.num_arms = len(action_to_price)

        self.reset()

    def _calc_ucb_range(self, j, i):
        # a helper to calculate the ucb in range j..i including j and i
        T_ji = np.sum(self.arm_to_num_pulls[j : i + 1])
        X_ji = np.sum((self.arm_to_mean_conversion[j : i + 1] * self.arm_to_num_pulls[j : i + 1])) / T_ji
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

        old_mean_conversion = self.arm_to_mean_conversion[action]
        old_count = self.arm_to_num_pulls[action]
        new_mean_conversion = (old_mean_conversion * old_count + conversion) / (old_count + 1)

        self.arm_to_mean_conversion[action] = new_mean_conversion
        self.arm_to_num_pulls[action] = old_count + 1

        self.explored_bandits += 1
        self.t += 1

    def reset(self):
        self.arm_to_mean_conversion = np.zeros(self.num_arms)
        self.arm_to_num_pulls = np.zeros(self.num_arms)

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round


class SLRAgent(Agent):
    def __init__(self, action_to_price, alpha=1):
        self.alpha = alpha
        self.action_to_price = action_to_price
        self.num_arms = len(action_to_price)
        self.slr = SimpleLinearRegressor()

        self.reset()

    def _choose_arm(self):
        if self.explored_bandits < self.num_arms:
            # not finished exploration phase
            k = self.explored_bandits
            return k

        estimated_quantities = self.slr.predict(x=np.array(self.action_to_price))
        means = self.action_to_price * estimated_quantities
        upper = np.sqrt(2 * np.log(self.t + 1) / (self.arm_to_num_pulls + 1))
        k = np.argmax(means + self.alpha * upper)
        # print(f"{means=}")
        # print(f"{estimated_quantities=}")
        # print(f"{self.arm_to_num_pulls=}")
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

        self.history_prices.append(price)
        self.history_quantities.append(conversion)
        # TODO: numpy conversion might be inefficient
        # change python list to numpy arrays initially
        self.slr.update_estimates(x=np.array([price]), y=np.array([conversion]))

        self.arm_to_num_pulls[action] += 1

        self.explored_bandits += 1
        self.t += 1

    def reset(self):
        self.history_prices = []
        self.history_quantities = []
        self.arm_to_num_pulls = np.zeros(self.num_arms)

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round


class AggregatingAgent(Agent):
    def __init__(
        self,
        action_to_price,
        aggregating_algorithm: AggregatingAlgorithm,
        alpha=1,
    ):
        self.alpha = alpha
        self.action_to_price = action_to_price
        self.num_arms = len(action_to_price)
        self.aggregating_algorithm = aggregating_algorithm

        self.reset()

    def _choose_arm(self):
        # if self.explored_bandits < self.num_arms:
        #     # not finished exploration phase
        #     k = self.explored_bandits
        #     return k

        estimated_quantities = self.aggregating_algorithm.predict(np.array(self.action_to_price))
        # print(f"{estimated_quantities=}")
        means = estimated_quantities
        upper = np.sqrt(2 * np.log(self.t + 1) / (self.arm_to_num_pulls + 1))
        k = np.argmax(self.action_to_price * (means + self.alpha * upper))

        # selected arm -> predict quantities for this arm
        _ = self.aggregating_algorithm.predict(np.array([self.action_to_price[k]]))

        # print(f"{means=}")
        # print(f"{estimated_quantities=}")
        # print(f"{self.arm_to_num_pulls=}")
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

        self.history_prices.append(price)
        self.history_quantities.append(conversion)
        self.aggregating_algorithm.update_estimates(x=np.array([price]), y=np.array([conversion]))

        self.arm_to_num_pulls[action] += 1

        self.explored_bandits += 1
        self.t += 1

    def reset(self):
        self.history_prices = []
        self.history_quantities = []
        self.arm_to_num_pulls = np.zeros(self.num_arms)

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round
