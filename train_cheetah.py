"""Train an agent for Roboschool Walker with VIME.
Soft actor-critic algorithm is used.
"""
from utils import lineplot, multiple_lineplot
import gym
import roboschool
import numpy as np
import torch
import toml
import tqdm
import argparse
from collections import namedtuple
import time
import shutil
import os

from agents.sac.agent import SAC
from memory import ReplayBuffer
from vime import VIME



ENV_NAME = "RoboschoolHalfCheetah-v1"




def train(config_file_path: str, save_dir: str, use_vime: bool, device: str):
    conf_d = toml.load(open(config_file_path))
    conf = namedtuple('Config', conf_d.keys())(*conf_d.values())

    # Check if saving directory is valid
    if "test" in save_dir and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if os.path.exists(save_dir):
        raise ValueError("Directory {} already exists.".format(save_dir))
    # Create save dir
    os.makedirs(save_dir)
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir)
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir)
    # Save config file
    shutil.copyfile(config_file_path, os.path.join(save_dir, os.path.basename(config_file_path)))

    # Set random variable
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.manual_seed(int(time.time()))

    # Set up log metrics
    metrics = {
        'epoch': [],
        'reward': [], # cummulated reward
        'curiosity_reward': [], # cummulated reward with information gain
        'D_KL_median': [], 'D_KL_mean': [],
        'q1_loss': [], 'policy_loss': [], 'alpha_loss': [], 'alpha': [],
        'ELBO': [],
    }

    # Set up environment
    env = gym.make(ENV_NAME)

    # Training set up
    agent = SAC(env.observation_space, env.action_space, device, **conf.agent)
    memory = ReplayBuffer(conf.replay_buffer_capacity, env.observation_space.shape, env.action_space.shape)
    vime = VIME(env.observation_space.shape[0], env.action_space.shape[0], **conf.vime) if use_vime else None
    # Load checkpoint if specified in config
    if conf.checkpoint != '':
        ckpt = torch.load(conf.checkpoint, map_location=device)
        metrics = ckpt['metrics']
        agent.load_state_dict(ckpt['agent'])
        memory.load_state_dict(ckpt['memory'])
        if use_vime:
            vime.load_state_dict(ckpt['vime'])

    def save_checkpoint():
        ckpt = {'metrics': metrics, 'agent': agent.state_dict(), 'memory': memory.state_dict()}
        if use_vime:
            ckpt['vime'] = vime.state_dict()
        path = os.path.join(ckpt_dir, 'checkpoint.pth')
        torch.save(ckpt, path)

    # Train agent
    init_epoch = 0 if len(metrics['epoch']) == 0 else metrics['epoch'][-1] + 1
    pbar = tqdm.tqdm(range(init_epoch, conf.epochs))
    reward_prev = None
    curiosity_reward_prev = None
    reward_moving_avg = None
    moving_avg_coef = 0.1
    for epoch in pbar:
        if reward_prev: 
            if use_vime:
                pbar.set_description("EPOCH {} --- Reward {:.2f}  Curiosity Reward {:.2f} (moving average {:.2f})".format(epoch, reward_prev, curiosity_reward_prev, reward_moving_avg))
            else:
                pbar.set_description("EPOCH {} --- Reward {:.2f} (moving average {:.2f})".format(epoch, reward_prev, reward_moving_avg))
        else:
            pbar.set_description("EPOCH {}".format(epoch))

        # Collect samples
        trajectory_samples, total_reward = collect_samples(env, agent, conf.episode_max_length)  # list of (s, a, r, s', t)
        reward_prev = total_reward
        reward_moving_avg = total_reward if reward_moving_avg is None else (1-moving_avg_coef) * reward_moving_avg + moving_avg_coef * total_reward
        assert len(trajectory_samples[0]) == 5 and isinstance(trajectory_samples[0], tuple), trajectory_samples[0]
        if use_vime:
            # Calculate D_KL to obtain surrogate reward r' = r + \eta D_KL
            info_gains = np.array([vime.calc_info_gain(s, a, s_next) for (s, a, _, s_next, _) in trajectory_samples])
            rewards_org = np.array([r for (_, _, r, _, _) in trajectory_samples])
            rewards = vime.calc_curiosity_reward(rewards_org, info_gains)
            curiosity_reward_prev = rewards.sum()
            vime.memorize_episodic_info_gains(info_gains)            
        else:
            rewards_org = [r for (_, _, r, _, _) in trajectory_samples]
            rewards = rewards_org

        metrics['epoch'].append(epoch)
        metrics['reward'].append(np.sum(rewards_org))
        metrics['curiosity_reward'].append(np.sum(rewards))
        lineplot(metrics['epoch'][-len(metrics['reward']):], metrics['reward'], 'reward', log_dir)
        lineplot(metrics['epoch'][-len(metrics['curiosity_reward']):], metrics['curiosity_reward'], 'curiosity_reward', log_dir)
        
        # Insert samples into the memory
        for (s, a, _, s_next, t), r in zip(trajectory_samples, rewards):
            memory.append(s, a, r, s_next, t)

        # Update parameters
        batch_data = memory.sample(conf.batch_size)
        q1_loss, q2_loss, policy_loss, alpha_loss, alphas = agent.update_parameters(batch_data, epoch)
        metrics['q1_loss'].append(q1_loss)
        metrics['policy_loss'].append(policy_loss)
        metrics['alpha_loss'].append(alpha_loss)
        metrics['alpha'].append(alphas)
        lineplot(metrics['epoch'][-len(metrics['q1_loss']):], metrics['q1_loss'], 'q1_loss', log_dir)
        lineplot(metrics['epoch'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', log_dir)
        lineplot(metrics['epoch'][-len(metrics['alpha_loss']):], metrics['alpha_loss'], 'alpha_loss', log_dir)
        lineplot(metrics['epoch'][-len(metrics['alpha']):], metrics['alpha'], 'alpha', log_dir)

        if use_vime:
            elbo = vime.update_posterior(memory)
            metrics['ELBO'].append(elbo)
            metrics['D_KL_median'].append(np.median(info_gains))
            metrics['D_KL_mean'].append(np.mean(info_gains))
            lineplot(metrics['epoch'][-len(metrics['ELBO']):], metrics['ELBO'], 'ELBO', log_dir)
            multiple_lineplot(metrics['epoch'][-len(metrics['D_KL_median']):], np.array([metrics['D_KL_median'], metrics['D_KL_mean']]).T, 'D_KL', ['median', 'mean'], log_dir)

        # Save checkpoint
        if epoch % conf.checkpoint_interval == 0:
            save_checkpoint()

    save_checkpoint()
    # Save the final model
    torch.save({'agent': agent.state_dict()}, os.path.join(ckpt_dir, 'final_model.pth'))
    



def evaluate(config_file_path: str, model_filepath: str, max_episode_length: int, render: bool):
    conf_d = toml.load(open(config_file_path))
    conf = namedtuple('Config', conf_d.keys())(*conf_d.values())

    # Set random variable
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    device = torch.device('cpu')

    env = gym.make(ENV_NAME)
    agent = SAC(env.observation_space, env.action_space, device, **conf.agent)
    ckpt = torch.load(model_filepath, map_location='cpu')
    agent.load_state_dict(ckpt['agent'])

    _, total_reward = collect_samples(env, agent, max_episode_length, evaluate=True, render=render)
    print("REWARD {}".format(total_reward))
    input("OK? >")




def collect_samples(env, agent, episode_max_length, evaluate=False, render=False):
    """Run for one episode using a given agent.
    Return
    ---
    trajectory: list of (s_t, a_t, r_t, s_{t+1}, terminate)
    total_reward: (float) Cumulated reward of an episode.
    """
    trajectory = []
    total_reward = 0

    o = env.reset()
    if render:
        env.render()
    for _ in range(episode_max_length):
        a = agent.select_action(o, eval=evaluate)
        o_next, r, done, _ = env.step(a)
        total_reward += r
        trajectory.append((o, a, r, o_next, done))
        if done:
            break
    return trajectory, total_reward


    
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trying a walker.')
    parser.add_argument('--config', default='default_config.toml', help='Config file path')
    parser.add_argument('--save-dir', default=os.path.join('results', 'test'), help='Save directory')
    parser.add_argument('--vime', action='store_true', help='Whether to use VIME.')
    parser.add_argument('--device', default='cpu', choices={'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'}, help='Device for computation.')
    parser.add_argument('-e', '--eval', action='store_true', help='Run model evaluation.')
    parser.add_argument('-m', '--model-filepath', default='', help='Path to trained model for evaluation.')
    parser.add_argument('-r', '--render', action='store_true', help='Render agent behavior during evaluation.')
    parser.add_argument('--max-episode-length', type=int, default=100000, help='Max episode duration for evaluation.')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.config, args.model_filepath, args.max_episode_length, args.render)
    else:
        train(args.config, args.save_dir, args.vime, args.device)
