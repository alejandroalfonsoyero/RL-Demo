import numpy as np
from matplotlib import pyplot as plt

from rl.agent.agent import Agent, ActionsSelectMethod, ActionsValueUpdateMethod
from rl.k_armed_test_bed.test_bed import KArmedTestBed


if __name__ == '__main__':

    x = np.random

    non_stationary_test_bed_10 = KArmedTestBed([-1, 2, 1.5, 2.2, -3.2, 0.5, -2.3, -3, 4.8, 1.3], 1,
                                               lambda r: r + 0.1 * (np.random.random() - 0.49))
    initial_values = {i: 0 for i in range(10)}
    steps = 1000

    simple_average_e_greedy = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                                    initial_values, exploration_factor=0.01, reward_average_window=100)
    weighted_average_e_greedy = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                                      initial_values, exploration_factor=0.01, reward_average_window=100,
                                      step_size=0.01)

    simple_average_e_greedy.run(steps, non_stationary_test_bed_10)
    weighted_average_e_greedy.run(steps, non_stationary_test_bed_10)

    simple_average_e_greedy.plot_average_reward(plt, "simple average e-greedy")
    weighted_average_e_greedy.plot_average_reward(plt, "weighted average e-greedy")

    plt.show()
