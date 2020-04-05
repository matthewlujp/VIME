# VIME
PyTorch implementation of "[VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)".  


### Install required packages with pip3
```sh
pip3 install -r requirements.txt
```

### Try out
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


### Implementation memos




### Evaluations
Performance of reinforcement learning with and without VIME was compared in the following environments.
* RoboschoolInvertedDoublePendulum-v1
* RoboschoolWalker2d-v1
* RoboschoolHumanoid-v1
* RoboschoolHalfCheetah-v1 (with sparse reward)

[Soft actor-critic (SAC)](https://arxiv.org/abs/1812.05905) is used as a base method.

##### RoboschoolInvertedDoublePendulum-v1
![RoboschoolInvertedDoublePendulum-v1]<img width="数値" alt="代替テキスト" src="">![RoboschoolWalker2d-v1]<img width="数値" alt="代替テキスト" src="">

##### RoboschoolWalker2d-v1


##### RoboschoolHumanoid-v1


##### RoboschoolHalfCheetah-v1 (with sparse reward)





