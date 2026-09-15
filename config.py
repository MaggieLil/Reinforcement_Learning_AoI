from dataclasses import dataclass


@dataclass
class EnvParams:
    """Parameters defining the Age of Information environment."""
    B_MAX: int = 5
    DELTA_MAX: int = 40
    P_E: float = 0.5
    P_SUCCESS: float = 0.9
    E_TX: int = 1

@dataclass
class ExperimentParams:
    """Parameters controlling the simulation experiments."""
    base_seed: int = 42
    num_runs: int = 10
    window_size: int = 2000
    results_dir: str = 'results'

EXPERIMENT_PARAMS = ExperimentParams()

@dataclass
class AgentParams:
        """Parameters controlling the reinforcement learning algorithms."""
        name: str = "DEFAULT"
        gamma_tau: float = 0.9995
        tau_start: float = 3.0
        steps: int = 20000
        epsilon: float = 0.1
        gamma_epsilon: float = 0.9997
        alpha_q: float = 0.05
        gamma_q: float = 0.9

LEARNING_PARAMS_SETS = [
    AgentParams(
        name="default"
    )
]
