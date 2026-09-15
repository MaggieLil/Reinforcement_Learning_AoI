import numpy as np

class SoftmaxPolicy:
    """Softmax action-selection policy for minimizing the estimated cost."""
    def __init__(self, tau, gamma_tau):
        self.tau = tau
        self.gamma_tau = gamma_tau

    def select_action(self, Q, battery_level, aoi, valid_actions):
        """Select an action according to Softmax probabilities."""
        q_values = np.array([Q[battery_level, aoi, a] for a in valid_actions])

        q_stable = q_values - np.min(q_values)

        scaled = -q_stable / max(self.tau, 1e-4)

        scaled = np.clip(scaled, -20, 20)

        exp_q = np.exp(scaled)
        softmax_policy = exp_q / np.sum(exp_q)

        return np.random.choice(valid_actions, p=softmax_policy)

    def decay(self):
        """Gradually reduce the exploration temperature."""
        self.tau *= self.gamma_tau

class GreedyPolicy:
    """Deterministic baseline that transmits whenever energy is available."""
    def __init__(self, env_params):
        self.energy_needed = env_params.E_TX

    def select_action(self, battery_level, valid_actions):
        """Select transmission whenever the required energy is available."""
        if 1 in valid_actions and battery_level >= self.energy_needed:
            return 1
        return 0

class EpsilonGreedyPolicy:
    """Epsilon-greedy action-selection policy."""
    def __init__(self, epsilon, gamma_epsilon, minimize=True):
        self.epsilon = epsilon
        self.gamma_epsilon = gamma_epsilon
        self.minimize = minimize

    def select_action(self, Q, battery_level, aoi, valid_actions):
        """Select a random action or the best estimated action."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = np.array([Q[battery_level, aoi, a] for a in valid_actions])

        if self.minimize:
            best_idx = np.argmin(q_values)
        else:
            best_idx = np.argmax(q_values)

        return valid_actions[best_idx]

    def decay(self):
        self.epsilon *= self.gamma_epsilon