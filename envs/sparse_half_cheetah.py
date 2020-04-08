import gym
from gym.spaces import Box
import roboschool
import numpy as np


class SparseHalfCheetah(gym.Env):
    metadata = {'render.modes': ['human', 'rgb']}

    def __init__(self, target_distance=5):
        super().__init__()
        self._target_distance = target_distance
        self._env = gym.make('RoboschoolHalfCheetah-v1')

    def _modified_observation(self, obs):
        return np.concatenate([obs, np.array(self._env.parts['torso'].pose().xyz())])

    def position(self):
        return self._env.parts['torso'].pose().xyz()

    def step(self, action):
        s, _, _, info = self._env.step(action)
        moved_distance = self._env.body_xyz[0] - self._env.start_pos_x
        r = float(moved_distance >= self._target_distance)
        return self._modified_observation(s), r, False, info
        
    def reset(self):
        s = self._env.reset()
        return self._modified_observation(s)

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.seed(seed)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return Box(
            low=np.concatenate([self._env.observation_space.low, np.ones(3) * -np.inf]),
            high=np.concatenate([self._env.observation_space.high, np.ones(3) * np.inf])
        )

    @property
    def reward_range(self):
        return self._env.reward_range