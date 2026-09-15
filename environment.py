import random

class AoIEnvironment:
    """Environment for an energy harvesting communication system."""
    def __init__(self, env_params):
        self.P_SUCCESS = env_params.P_SUCCESS
        self.P_E = env_params.P_E
        self.B_MAX = env_params.B_MAX
        self.DELTA_MAX = env_params.DELTA_MAX
        self.energy_needed = env_params.E_TX

        self.battery_level = 0
        self.aoi = 1

    def reset(self):
        """Reset the battery level and AoI to their initial values."""
        self.battery_level = 0
        self.aoi = 1
        return self.battery_level, self.aoi

    def get_valid_actions(self):
        """Return actions that are feasible in the current state."""
        valid_actions = [0]
        if self.battery_level >= self.energy_needed:
            valid_actions.append(1)
        return valid_actions

    def step(self, action):
        """Execute an action and return the AoI cost and next state."""
        aoi_cost = self.aoi

        if action == 1 and self.battery_level >= self.energy_needed:
            self.battery_level -= self.energy_needed
            if random.random() < self.P_SUCCESS:
                next_aoi = 1
            else:
                next_aoi = min(self.aoi + 1, self.DELTA_MAX)
        else:
            next_aoi = min(self.aoi + 1, self.DELTA_MAX)

        if random.random() < self.P_E:
            next_battery_level = min(self.battery_level + 1, self.B_MAX)
        else:
            next_battery_level = self.battery_level

        self.battery_level = next_battery_level
        self.aoi = next_aoi

        return aoi_cost, (self.battery_level, self.aoi)