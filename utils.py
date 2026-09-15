import os
import shutil
import csv


def prepare_results_dir(results_dir):
    """Remove the existing results directory and create a clean one."""
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)


def create_experiment_dir(results_dir, experiment_name):
    """Create and return a directory for one experiment configuration."""
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def create_csv(filename):
    """Create a CSV file and write its header."""
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Run",
            "Step",
            "Algorithm",
            "Policy",
            "J",
            "Avg_cost",
            "Tau",
            "Epsilon"
        ])


def save_config(filename, env_params, experiment_params, agent_params):
    """Save environment, experiment, and agent parameters to a CSV file."""
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow([
            "Section",
            "Parameter",
            "Value"
        ])

        for k, v in vars(env_params).items():
            writer.writerow(["Environment", k, v])

        for k, v in vars(experiment_params).items():
            writer.writerow(["Experiment", k, v])

        for k, v in vars(agent_params).items():
            writer.writerow(["Agent", k, v])