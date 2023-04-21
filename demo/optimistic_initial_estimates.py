from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

from rl.agent.agent import Agent, ActionsSelectMethod, ActionsValueUpdateMethod
from rl.k_armed_test_bed.test_bed import KArmedTestBed


if __name__ == '__main__':

    test_bed_10 = KArmedTestBed([-1, 2, 1.5, 2.2, -3.2, 0.5, -2.3, -3, 4.8, 1.3], 2)
    realistic_initial_values = {i: 0 for i in range(10)}
    optimistic_initial_values = {i: 8 for i in range(10)}
    steps = 10000
    fig, (ax, ax2) = subplots(2)

    optimistic_greedy = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                              optimistic_initial_values, exploration_factor=0, reward_average_window=100)
    realistic_e_greedy_001 = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                                   realistic_initial_values, exploration_factor=0.01, reward_average_window=100)

    optimistic_greedy.run(steps, test_bed_10)
    realistic_e_greedy_001.run(steps, test_bed_10)

    optimistic_greedy.plot_average_reward(ax, "optimistic greedy")
    realistic_e_greedy_001.plot_average_reward(ax, "realistic e-greedy (e=0.001)")

    plt.show()
