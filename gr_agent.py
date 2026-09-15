import numpy as np

from agent import Agent

class GRAgent(Agent):
    """Average-cost reinforcement learning agent."""
    def __init__(self, env_params, policy):
        super().__init__(env_params, policy)
        self.J = 0.0
        self.m_counts = np.ones_like(self.Q)
        self.n_steps = 1

    def update(self, state, action, reward, next_state, valid_next_actions):
        """Update the Q-values and average-cost estimate."""
        aoi_cost = -reward
        battery_level, aoi = state
        next_battery_level, next_aoi = next_state

        best_q_next = np.min([
            self.Q[next_battery_level, next_aoi, a]
            for a in valid_next_actions]
        )

        alpha = 1.0 / np.sqrt(self.m_counts[battery_level, aoi, action])
        beta = 1.0 / self.n_steps

        td_error = aoi_cost - self.J + best_q_next - self.Q[battery_level, aoi, action]
        self.Q[battery_level, aoi, action] += alpha * td_error

        self.J = self.J + beta * (aoi_cost - self.J)

        self.policy.decay()
        self.m_counts[battery_level, aoi, action] += 1
        self.n_steps += 1