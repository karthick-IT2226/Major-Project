import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MentalHealthEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(5, dtype=np.float32), 0.0, True, False, {}
