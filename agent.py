import numpy as np

class Agent:
    """Base class for Q-table-based reinforcement learning agents."""
    def __init__(self, env_params, policy):
        self.policy = policy
        self.Q = np.zeros((env_params.B_MAX + 1, env_params.DELTA_MAX + 1, 2))

    def select_action(self, state, valid_actions):
        """Select an action using the current policy."""
        battery_level, aoi = state
        return self.policy.select_action(self.Q, battery_level, aoi, valid_actions)

    def update(self, state, action, reward, next_state, valid_next_actions):
        """Update the agent's learned values."""
        raise NotImplementedError