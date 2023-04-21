from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

from rl.agent.agent import Agent, ActionsSelectMethod, ActionsValueUpdateMethod
from rl.k_armed_test_bed.test_bed import KArmedTestBed


if __name__ == '__main__':

    test_bed_10 = KArmedTestBed([-1, 2, 1.5, 2.2, -3.2, 0.5, -2.3, -3, 4.8, 1.3], 2)
    initial_values = {i: 0 for i in range(10)}
    steps = 1000
    fig, (ax, ax2) = subplots(2)

    greedy = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE, initial_values,
                   exploration_factor=0, reward_average_window=100)
    e_greedy_0001 = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                          initial_values, exploration_factor=0.001, reward_average_window=100)
    e_greedy_001 = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                         initial_values, exploration_factor=0.01, reward_average_window=100)
    e_greedy_01 = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                        initial_values, exploration_factor=0.1, reward_average_window=100)
    e_greedy_05 = Agent(ActionsSelectMethod.E_GREEDY, ActionsValueUpdateMethod.SIMPLE_AVERAGE,
                        initial_values, exploration_factor=0.5, reward_average_window=100)

    greedy.run(steps, test_bed_10)
    e_greedy_0001.run(steps, test_bed_10)
    e_greedy_001.run(steps, test_bed_10)
    e_greedy_01.run(steps, test_bed_10)
    e_greedy_05.run(steps, test_bed_10)

    greedy.plot_average_reward(ax, "greedy")
    e_greedy_0001.plot_average_reward(ax, "e-greedy (e=0.001)")
    e_greedy_001.plot_average_reward(ax, "e-greedy (e=0.01)")
    e_greedy_01.plot_average_reward(ax, "e-greedy (e=0.1)")
    e_greedy_05.plot_average_reward(ax, "e-greedy (e=0.5)")

    plt.show()
