from typing import List

import numpy as np


class Bandit:

    def __init__(self, uid: int, expected_reward: float, std: float = 1, name: str = None):
        self.uid = uid
        self.name = name
        self.expected = expected_reward
        self.std = std

    def action_value(self):
        return np.random.normal(self.expected, self.std)


class KArmedTestBed:

    def __init__(self, expected_rewards: List[int], std: float = 1):
        self.bandits = {}
        for index, rew in zip(range(len(expected_rewards)), expected_rewards):
            self.bandits.update({index: Bandit(index, rew, std, f"Bandit-{index}")})

    def action_value(self, bandit_uid: int):
        return self.bandits[bandit_uid].action_value()
