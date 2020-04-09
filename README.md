# VIME
PyTorch implementation of "[VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)".  


## Install required packages with pip3
```sh
pip3 install -r requirements.txt
```

## Try out
Train agent in "RoboschoolHalfCheetah-v1".
A dedicated configuration file for this environment is in configs directory.
```sh
python3 main.py --config configs/config_cheetah.toml --save-dir results/test_cheetah --vime
```

A model file is saved in ```results/test_cheetah/checkpoints/model.pth```.
Load this file to run the trained policy.
```sh
python3 main.py --config configs/config_cheetah.toml -m results/test_cheetah/checkpoints/model.pth -e -r
```

Options
```
--config              Config file path
--save-dir            Save directory
--vime                Whether to use VIME.
--visualize-interval  Interval to draw graphs of metrics.
--device              Device for computation.
-e, --eval            Run model evaluation.
-m, --model-filepath  Path to trained model for evaluation.
-r, --render          Render agent behavior during evaluation.
```


## Implementation notes
Explanation of implementation is in [Implementation notes of "VIME: Variational Information Maximizing Exploration"](https://qiita.com/matthewlujp/items/84ffa27ab63ac9800824).



## Evaluations
Performance of reinforcement learning with and without VIME was compared in the following environments.
* RoboschoolInvertedDoublePendulum-v1
* RoboschoolWalker2d-v1
* RoboschoolHumanoid-v1
* RoboschoolHalfCheetah-v1 (with sparse reward)

In HalfCheetah, +1.0 was provided as a reward when a body moved more than 5 units.
[Soft actor-critic (SAC)](https://arxiv.org/abs/1812.05905) is used as a base method.

|InvertedDoublePendulum-v1|Walker2d-v1|
|---|---|
|![](https://user-images.githubusercontent.com/13263381/78898710-35e09e80-7aaf-11ea-8190-e08189f42e08.png)|![](https://user-images.githubusercontent.com/13263381/78898716-37aa6200-7aaf-11ea-9f21-a4ef0ce8c4e6.png)|

|Humanoid-v1|HalfCheetah-v1 (sparse reward)|
|---|---|
|![](https://user-images.githubusercontent.com/13263381/78898714-3711cb80-7aaf-11ea-85c1-4a560ebb8604.png)|![](https://user-images.githubusercontent.com/13263381/78898715-3711cb80-7aaf-11ea-8b6e-525f74e76e08.png)|


##### Note on modifying environment
An instance of RoboschoolHalfCheetah-v1 holds body position to calculate a reward.
The information can be accessed by ```env.body_xyz[0]``` (see [gym_forward_walker.py](https://github.com/openai/roboschool/blob/master/roboschool/gym_forward_walker.py) for details).  

RoboschoolHalfCheetah-v1 was wraped in a new environment "SparseHalfCheetah", which returns a reward of +1.0 when a body moves more than 5 units. 
To create a new custom environment, define a class which inherits ```gym.Env``` and implement follow methods
* reset
* step
* render
* close
* seed

and following properties
* action_space
* observation_space
* reward_range

By registering the newly defined environment, you can instantiate the environment using ```gym.make``` method.
```python
from gym.envs.registration import register
register(
    id='EnvironmentName-v1',
    entry_point=NewEnvironmentClass,
    max_episode_steps=1000,
    reward_threshold=1,
    kwargs={},
)
```

You can provide keyword arguments to the new class through kwargs.
For further information, refer to [registration.py](https://github.com/openai/gym/blob/master/gym/envs/registration.py).





