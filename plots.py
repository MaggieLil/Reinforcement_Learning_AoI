import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_all_alg_cumulative(
        mean_rl_cumulative,
        mean_greedy_cumulative,
        mean_epsilon_cumulative,
        mean_q_cumulative,
        steps,
        save_path
):
    """Plot cumulative average AoI for all evaluated algorithms."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mean_rl_cumulative,
        label="GR-learning with Softmax",
        color="red",
        linewidth=2
    )
    plt.plot(
        mean_greedy_cumulative,
        label="Greedy baseline",
        color="green",
        linestyle="--"
    )
    plt.plot(
        mean_epsilon_cumulative,
        label="GR-learning with Epsilon-greedy",
        color="blue",
        linestyle="--"
    )
    plt.plot(
        mean_q_cumulative,
        label="Q-learning",
        color="black",
        linestyle=":"
    )

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Average AoI (cumulative)", fontsize=12)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.title("Algorithms Performance")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    plt.xlim(0, steps)
    plt.ylim(1.5, 6)

    plt.savefig(save_path)
    plt.close()

# =========================

def plot_all_alg_moving(
        mean_rl_moving,
        mean_greedy_moving,
        mean_epsilon_moving,
        mean_q_moving,
        steps,
        save_path
):
    """Plot moving-average AoI for all evaluated algorithms."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mean_rl_moving,
        label="GR-learning with Softmax",
        color="red",
        linewidth=2
    )
    plt.plot(
        mean_greedy_moving,
        label="Greedy baseline",
        color="green",
        linestyle="--"
    )
    plt.plot(
        mean_epsilon_moving,
        label="GR-learning with Epsilon-greedy",
        color="blue",
        linestyle="--"
    )
    plt.plot(
        mean_q_moving,
        label="Q-learning",
        color="black",
        linestyle=":"
    )

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Average AoI (moving)", fontsize=12)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.title("Algorithms Performance")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    plt.xlim(0, steps)
    plt.ylim(1.5, 6)

    plt.savefig(save_path)
    plt.close()

# =========================

def plot_gr_softmax_vs_greedy(
        mean_rl_cumulative,
        mean_greedy_cumulative,
        steps,
        save_path
):
    """Compare cumulative AoI of GR-learning with Softmax and the greedy baseline."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mean_rl_cumulative,
        label="GR-learning with Softmax",
        linewidth=2,
        color="red"
    )

    plt.plot(
        mean_greedy_cumulative,
        label="Greedy baseline",
        linestyle="--",
        color="green"
    )

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Average AoI", fontsize=12)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.title("GR-learning with Softmax vs Greedy baseline")
    plt.legend(fontsize=11)
    plt.xlim(0, steps)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(save_path)
    plt.close()

# =========================

def plot_gr_softmax_vs_epsilon(
        mean_rl_cumulative,
        mean_epsilon_cumulative,
        steps,
        save_path
):
    """Compare GR-learning with Softmax and epsilon-greedy exploration."""
    plt.figure(figsize=(10, 6))

    plt.plot(
        mean_rl_cumulative,
        label="GR-learning with Softmax",
        linewidth=2,
        color="red",
    )

    plt.plot(
        mean_epsilon_cumulative,
        label="GR-learning with Epsilon-greedy",
        color="blue",
        linestyle="--"
    )

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Average AoI", fontsize=12)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.title("GR - Softmax policy vs Epsilon-greedy policy")
    plt.legend(fontsize=11)
    plt.xlim(0, steps)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(save_path)
    plt.close()