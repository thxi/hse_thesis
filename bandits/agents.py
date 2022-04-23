from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass

    def update_estimates(self, action, observation, reward):
        pass


class EpsilonGreedyAgent(Agent):
    def __init__(self, eps, num_arms):
        # the probability of selecting a random arm
        self.eps = eps
        self.num_arms = num_arms
        self.reset()

    def _choose_arm(self):
        should_explore = np.random.binomial(1, self.eps)
        if should_explore == 1:
            # select random
            k = np.random.randint(0, self.num_arms)
        else:
            # select best
            k = np.argmax(self.arm_to_mean_rewards)
        return k

    def get_action(self, observation):
        action = self._choose_arm()
        return action

    def update_estimates(self, action, observation, reward):
        old_mean_reward = self.arm_to_mean_rewards[action]
        old_count = self.arm_to_num_pulls[action]
        new_mean_reward = (old_mean_reward * old_count + reward) / (old_count + 1)

        self.arm_to_mean_rewards[action] = new_mean_reward
        self.arm_to_num_pulls[action] = old_count + 1

    def reset(self):
        self.arm_to_mean_rewards = np.zeros(self.num_arms)
        self.arm_to_num_pulls = np.zeros(self.num_arms)


class UCB1Agent(Agent):
    def __init__(self, num_arms, alpha=1, reward_normalization=1):
        self.num_arms = num_arms
        self.alpha = alpha
        self.reward_normalization = reward_normalization

        self.reset()

    def _choose_arm(self):
        if self.explored_bandits < self.num_arms:
            # not finished exploration phase
            k = self.explored_bandits
            return k

        # selecting based on upper confidence bound
        means = self.arm_to_mean_rewards
        upper = np.sqrt(2 * np.log(self.t + 1) / (self.arm_to_num_pulls + 1))
        k = np.argmax(means + self.alpha * upper)
        return k

    def get_action(self, observation):
        action = self._choose_arm()
        return action

    def update_estimates(self, action, observation, reward):
        reward = reward / self.reward_normalization
        old_mean_reward = self.arm_to_mean_rewards[action]
        old_count = self.arm_to_num_pulls[action]
        new_mean_reward = (old_mean_reward * old_count + reward) / (old_count + 1)

        self.arm_to_mean_rewards[action] = new_mean_reward
        self.arm_to_num_pulls[action] = old_count + 1

        self.explored_bandits += 1
        self.t += 1

    def reset(self):
        self.arm_to_mean_rewards = np.zeros(self.num_arms)
        self.arm_to_num_pulls = np.zeros(self.num_arms)

        # firstly, we should explore all bandit arms
        # so explored_bandits shows the number of bandits we explored so far
        self.explored_bandits = 0
        self.t = 0  # current round


class ThompsonSamplingBetaAgent(Agent):
    def __init__(self, num_arms):
        self.num_arms = num_arms

        self.reset()

    def _choose_arm(self):
        alpha = self.arm_to_mean_rewards * self.arm_to_num_pulls
        beta = self.arm_to_num_pulls - alpha
        theta = np.random.beta(alpha + 1, beta + 1)  # sample from beta dist for each arm
        k = np.argmax(theta)
        return k

    def get_action(self, observation):
        action = self._choose_arm()
        return action

    def update_estimates(self, action, observation, reward):
        old_mean_reward = self.arm_to_mean_rewards[action]
        old_count = self.arm_to_num_pulls[action]
        new_mean_reward = (old_mean_reward * old_count + reward) / (old_count + 1)

        self.arm_to_mean_rewards[action] = new_mean_reward
        self.arm_to_num_pulls[action] = old_count + 1

    def reset(self):
        self.arm_to_mean_rewards = np.zeros(self.num_arms)
        self.arm_to_num_pulls = np.zeros(self.num_arms)
