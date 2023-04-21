from typing import List

import numpy as np


class Bandit:

    def __init__(self, uid: int, expected_reward: float, std: float = 1, name: str = None):
        self.uid = uid
        self.name = name
        self.expected_reward = expected_reward
        self.std = std

    def action_value(self):
        return np.random.normal(self.expected_reward, self.std)


class KArmedTestBed:

    def __init__(self, expected_rewards: List[int], std: float = 1, mean_reward_variation: callable = None):
        self.bandits = {}
        self.mean_reward_variation = mean_reward_variation

        for index, rew in zip(range(len(expected_rewards)), expected_rewards):
            self.bandits.update({index: Bandit(index, rew, std, f"Bandit-{index}")})

    def action_value(self, bandit_uid: int):

        bandit = self.bandits[bandit_uid]

        # Non-Stationary environment
        if callable(self.mean_reward_variation):
            bandit.expected_reward = self.mean_reward_variation(bandit.expected_reward)
        # Stationary environment
        else:
            pass

        return bandit.action_value()
