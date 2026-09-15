import numpy as np

from agent import Agent

class QLearningAgent(Agent):
    """Q-learning agent using discounted reward maximization."""
    def __init__(self, env_params, policy, alpha, gamma):
        super().__init__(env_params, policy)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state, action, reward, next_state, valid_next_actions):
        """Update the Q-value using the Q-learning rule."""
        battery_level, aoi = state
        next_battery_level, next_aoi = next_state

        # Select the highest estimated value among feasible next actions.
        best_next = np.max([
            self.Q[next_battery_level, next_aoi, a]
            for a in valid_next_actions
        ])

        # Compute the discounted Q-learning target.
        td_target = reward + self.gamma * best_next

        # Temporal-difference error.
        td_error = (td_target - self.Q[battery_level, aoi, action])

        # Update the current state-action value.
        self.Q[battery_level, aoi, action] += (self.alpha * td_error)

        self.policy.decay()