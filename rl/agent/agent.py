from collections import deque
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

from rl.k_armed_test_bed.test_bed import KArmedTestBed


class ActionsSelectMethod(Enum):
    E_GREEDY = "e-greedy"


class ActionsValueUpdateMethod(Enum):
    SIMPLE_AVERAGE = "simple-average"
    WEIGHTED_AVERAGE = "weighted-average"


class Agent:

    def __init__(self, action_select_method: ActionsSelectMethod, action_value_update_method: ActionsValueUpdateMethod,
                 action_q_values: dict, exploration_factor: float = 0, reward_average_window: int = 10,
                 step_size: float = 1):
        self.action_q_values = action_q_values
        self.action_select_method = action_select_method
        self.action_value_update_method = action_value_update_method
        self.exploration_factor = exploration_factor
        self.step_size = step_size
        self.last_rewards = deque(maxlen=reward_average_window)
        self.average_reward = [0]
        self.steps_hist = []

        self.selected_count = {key: 0 for key in self.action_q_values}

    def pick_e_greedy(self):
        explore = np.random.uniform() < self.exploration_factor
        max_estimate = max(self.action_q_values.values())
        greedy_actions = [k for k, v in self.action_q_values.items() if v == max_estimate]
        non_greedy_actions = [k for k, v in self.action_q_values.items() if v != max_estimate]

        if explore:
            return np.random.choice(non_greedy_actions)
        else:
            return np.random.choice(greedy_actions)

    def update_q_table(self, selected_action: int, reward):
        if self.action_value_update_method == ActionsValueUpdateMethod.SIMPLE_AVERAGE:
            self.selected_count[selected_action] += 1
            q_n = self.action_q_values[selected_action]
            n = self.selected_count[selected_action]
            self.action_q_values[selected_action] = q_n + (1 / n) * (reward - q_n)
        elif self.action_value_update_method == ActionsValueUpdateMethod.WEIGHTED_AVERAGE:
            q_n = self.action_q_values[selected_action]
            self.action_q_values[selected_action] = q_n + self.step_size * (reward - q_n)

    def run(self, steps: int, test_bed: KArmedTestBed):
        self.steps_hist.clear()
        self.average_reward.clear()
        self.last_rewards.clear()

        for i in range(steps):
            self.steps_hist.append(i)
            if self.action_select_method == ActionsSelectMethod.E_GREEDY:
                action = self.pick_e_greedy()
                reward = test_bed.action_value(action)
                self.last_rewards.append(reward)
                self.average_reward.append(sum(self.last_rewards) / len(self.last_rewards))
                self.update_q_table(action, reward)

    def plot_average_reward(self, axis, label):
        axis.plot(self.steps_hist, self.average_reward, label=label)
        if axis == plt:
            axis.xlabel("steps")
            axis.ylabel("average reward")
        else:
            axis.set_xlabel("steps")
            axis.set_ylabel("average reward")
        axis.legend()
