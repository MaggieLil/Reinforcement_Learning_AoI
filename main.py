import random
import numpy as np
import csv
import os

from config import (
    EnvParams,
    LEARNING_PARAMS_SETS,
    EXPERIMENT_PARAMS
)
from policy import (
    SoftmaxPolicy,
    GreedyPolicy,
    EpsilonGreedyPolicy
)
from environment import AoIEnvironment
from gr_agent import GRAgent
from q_agent import QLearningAgent

from plots import (
    plot_all_alg_cumulative,
    plot_all_alg_moving,
    plot_gr_softmax_vs_greedy,
    plot_gr_softmax_vs_epsilon
)

from utils import (
    prepare_results_dir,
    create_experiment_dir,
    create_csv,
    save_config
)

def run_gr_softmax_simulation(env_params, l_params, steps, window_size, run_idx, csv_filename):
    """Run one GR-learning experiment using Softmax exploration."""
    env = AoIEnvironment(env_params)
    policy = SoftmaxPolicy(l_params.tau_start, l_params.gamma_tau)
    agent = GRAgent(env_params, policy)
    state = env.reset()

    costs = []
    cumulative_curves = []
    moving_curves = []
    total_aoi = 0

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        for n in range(1, steps + 1):
            # Select an action that is feasible in the current state.
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            # Execute the action and observe the resulting AoI cost and state.
            aoi_cost, next_state = env.step(action)

            costs.append(aoi_cost)
            total_aoi += aoi_cost

            # Cumulative average AoI over all time steps so far.
            cumulative_avg = total_aoi / n
            cumulative_curves.append(cumulative_avg)

            # Moving average AoI over the most recent time window.
            start_idx = max(0, len(costs) - window_size)
            moving_avg = np.mean(costs[start_idx:])
            moving_curves.append(moving_avg)

            # Determine which actions are feasible in the next state.
            valid_next_actions = env.get_valid_actions()

            # RL agents maximize reward, so AoI cost is converted to negative reward.
            reward = -aoi_cost

            agent.update(state, action, reward, next_state, valid_next_actions)
            state = next_state

            # Log progress approximately ten times per run.
            log_interval = max(1, steps // 10)
            if n % log_interval == 0:
                writer.writerow([
                    run_idx,
                    n,
                    'gr-learning',
                    'softmax',
                    f"{agent.J:.3f}",
                    "-",
                    f"{policy.tau:.2e}",
                    "-"
                ])

    return cumulative_curves, moving_curves

def run_gr_epsilon_simulation(env_params, l_params, steps, window_size, run_idx, csv_filename):
    """Run one GR-learning experiment using epsilon-greedy exploration."""
    env = AoIEnvironment(env_params)
    policy = EpsilonGreedyPolicy(
        l_params.epsilon,
        l_params.gamma_epsilon,
        minimize=True
    )
    agent = GRAgent(env_params, policy)
    state = env.reset()

    costs = []
    cumulative_curves = []
    moving_curves = []
    total_aoi = 0

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        for n in range(1, steps + 1):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            aoi_cost, next_state = env.step(action)

            costs.append(aoi_cost)
            total_aoi += aoi_cost

            cumulative_avg = total_aoi / n
            cumulative_curves.append(cumulative_avg)

            start_idx = max(0, len(costs) - window_size)
            moving_avg = np.mean(costs[start_idx:])
            moving_curves.append(moving_avg)

            valid_next_actions = env.get_valid_actions()

            reward = -aoi_cost
            agent.update(state, action, reward, next_state, valid_next_actions)
            state = next_state

            log_interval = max(1, steps // 10)
            if n % log_interval == 0:
                writer.writerow([
                    run_idx,
                    n,
                    'gr-learning',
                    'epsilon',
                    f"{agent.J:.3f}",
                    "-",
                    "-",
                    f"{policy.epsilon:.2e}"])

    return cumulative_curves, moving_curves

def run_q_epsilon_simulation(env_params, l_params, steps, window_size, run_idx, csv_filename):
    """Run one Q-learning experiment using epsilon-greedy exploration."""
    env = AoIEnvironment(env_params)
    policy = EpsilonGreedyPolicy(
        l_params.epsilon,
        l_params.gamma_epsilon,
        minimize=False
    )
    agent = QLearningAgent(
        env_params,
        policy,
        l_params.alpha_q,
        l_params.gamma_q
    )
    state = env.reset()

    costs = []
    cumulative_curves = []
    moving_curves = []
    total_aoi = 0

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        for n in range(1, steps + 1):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            aoi_cost, next_state = env.step(action)

            costs.append(aoi_cost)
            total_aoi += aoi_cost

            cumulative_avg = total_aoi / n
            cumulative_curves.append(cumulative_avg)

            start_idx = max(0, len(costs) - window_size)
            moving_avg = np.mean(costs[start_idx:])
            moving_curves.append(moving_avg)

            valid_next_actions = env.get_valid_actions()

            reward = -aoi_cost
            agent.update(
                state,
                action,
                reward,
                next_state,
                valid_next_actions
            )
            state = next_state

            log_interval = max(1, steps // 10)
            if n % log_interval == 0:
                writer.writerow([
                    run_idx,
                    n,
                    'q-learning',
                    'epsilon',
                    "-",
                    f"{np.mean(costs):.3f}",
                    "-",
                    f"{policy.epsilon:.2e}"])

    return cumulative_curves, moving_curves

def run_greedy_simulation(env_params, steps, window_size):
    """Run the greedy baseline without reinforcement learning."""
    policy_greedy = GreedyPolicy(env_params)
    env_greedy = AoIEnvironment(env_params)
    battery_level, aoi = env_greedy.reset()

    greedy_costs = []
    greedy_cumulative = []
    greedy_moving = []
    greedy_total_aoi = 0

    for n in range(1, steps + 1):
        valid_actions = env_greedy.get_valid_actions()
        action = policy_greedy.select_action(battery_level, valid_actions)

        aoi_cost, next_state = env_greedy.step(action)

        greedy_costs.append(aoi_cost)
        greedy_total_aoi += aoi_cost

        greedy_cumulative.append(greedy_total_aoi / n)
        start_idx = max(0, len(greedy_costs) - window_size)
        greedy_moving.append(np.mean(greedy_costs[start_idx:]))

        battery_level, aoi = next_state

    return greedy_cumulative, greedy_moving


base_seed = EXPERIMENT_PARAMS.base_seed
num_runs = EXPERIMENT_PARAMS.num_runs
window_size = EXPERIMENT_PARAMS.window_size

prepare_results_dir(EXPERIMENT_PARAMS.results_dir)

env_params = EnvParams()

for idx, l_params in enumerate(LEARNING_PARAMS_SETS):

    experiment_dir = create_experiment_dir(
        EXPERIMENT_PARAMS.results_dir,
        l_params.name
    )

    save_config(
        os.path.join(experiment_dir, "config.csv"),
        env_params,
        EXPERIMENT_PARAMS,
        l_params
    )


    print("=" * 60)
    print(f"EXPERIMENT: {l_params.name} ({idx+1}/{len(LEARNING_PARAMS_SETS)})")
    print("=" * 60)

    steps = l_params.steps

    csv_softmax = os.path.join(
      experiment_dir,
      "gr_softmax.csv"
    )

    csv_epsilon = os.path.join(
        experiment_dir,
        "gr_epsilon.csv"
    )

    csv_qlearning = os.path.join(
        experiment_dir,
        "q_learning.csv"
    )

    create_csv(csv_softmax)
    create_csv(csv_epsilon)
    create_csv(csv_qlearning)


    all_gr_softmax_cumulative_curves = []
    all_gr_softmax_moving_curves = []
    all_greedy_cumulative_curves = []
    all_greedy_moving_curves = []
    all_gr_epsilon_cumulative_curves = []
    all_gr_epsilon_moving_curves = []
    all_q_cumulative_curves = []
    all_q_moving_curves = []

    for run in range(num_runs):

        current_seed = base_seed + run
        print(f"\n===== URUCHOMIENIE (RUN) {run + 1}/{num_runs} (Seed: {current_seed}) =====")

        # -----------------------------------------------------
        # GR-LEARNING WITH SOFTMAX
        # -----------------------------------------------------
        # Use the same random seed for all algorithms within a run
        # to ensure a fair comparison of the stochastic environment.
        np.random.seed(current_seed)
        random.seed(current_seed)

        gr_cumulative, gr_moving = run_gr_softmax_simulation(
          env_params,
          l_params,
          steps,
          window_size,
          run + 1,
          csv_softmax
        )
        all_gr_softmax_cumulative_curves.append(gr_cumulative)
        all_gr_softmax_moving_curves.append(gr_moving)

        # -----------------------------------------------------
        # GREEDY BASELINE
        # -----------------------------------------------------
        np.random.seed(current_seed)
        random.seed(current_seed)

        greedy_cumulative, greedy_moving = run_greedy_simulation(
            env_params,
            steps,
            window_size
        )
        all_greedy_cumulative_curves.append(greedy_cumulative)
        all_greedy_moving_curves.append(greedy_moving)

        # -----------------------------------------------------
        # GR-LEARNING WITH EPSILON-GREEDY
        # -----------------------------------------------------
        np.random.seed(current_seed)
        random.seed(current_seed)

        gr_eps_cumulative, gr_eps_moving = run_gr_epsilon_simulation(
          env_params,
          l_params,
          steps,
          window_size,
          run + 1,
          csv_epsilon
        )
        all_gr_epsilon_cumulative_curves.append(gr_eps_cumulative)
        all_gr_epsilon_moving_curves.append(gr_eps_moving)

        # -----------------------------------------------------
        # Q-LEARNING WITH EPSILON-GREEDY
        # -----------------------------------------------------
        np.random.seed(current_seed)
        random.seed(current_seed)

        q_cumulative, q_moving = run_q_epsilon_simulation(
            env_params,
            l_params,
            steps,
            window_size,
            run + 1,
            csv_qlearning
        )
        all_q_cumulative_curves.append(q_cumulative)
        all_q_moving_curves.append(q_moving)

    # Average the learning curves across all independent runs.
    mean_rl_cumulative = np.mean(all_gr_softmax_cumulative_curves, axis=0)
    mean_rl_moving = np.mean(all_gr_softmax_moving_curves, axis=0)
    mean_greedy_cumulative = np.mean(all_greedy_cumulative_curves, axis=0)
    mean_greedy_moving = np.mean(all_greedy_moving_curves, axis=0)
    mean_epsilon_cumulative = np.mean(all_gr_epsilon_cumulative_curves, axis=0)
    mean_epsilon_moving = np.mean(all_gr_epsilon_moving_curves, axis=0)
    mean_q_cumulative = np.mean(all_q_cumulative_curves, axis = 0)
    mean_q_moving = np.mean(all_q_moving_curves, axis = 0)

    plot_all_alg_cumulative(
        mean_rl_cumulative,
        mean_greedy_cumulative,
        mean_epsilon_cumulative,
        mean_q_cumulative,
        steps,
        os.path.join(
            experiment_dir,
            "comparison_cumulative.png"
        )
    )

    plot_all_alg_moving(
        mean_rl_moving,
        mean_greedy_moving,
        mean_epsilon_moving,
        mean_q_moving,
        steps,
        os.path.join(
            experiment_dir,
            "comparison_moving.png"
        )
    )

    plot_gr_softmax_vs_greedy(
        mean_rl_cumulative,
        mean_greedy_cumulative,
        steps,
        os.path.join(
            experiment_dir,
            "gr_softmax_vs_greedy.png"
        )
    )

    plot_gr_softmax_vs_epsilon(
        mean_rl_cumulative,
        mean_epsilon_cumulative,
        steps,
        os.path.join(
            experiment_dir,
            "gr_softmax_vs_epsilon.png"
        )
    )