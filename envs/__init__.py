from gym.envs.registration import register
from .sparse_half_cheetah import SparseHalfCheetah

register(
    id='SparseHalfCheetah-v1',
    entry_point=SparseHalfCheetah,
    max_episode_steps=1000,
    kwargs={'target_distance': 5},
)