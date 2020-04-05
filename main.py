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
import envs  # custom environments


def train(config_file_path: str, save_dir: str, use_vime: bool, device: str, visualize_interval: int):
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
        'episode': [],
        'collected_samples': [],
        'reward': [], # cummulated reward
        'curiosity_reward': [], # cummulated reward with information gain
        'likelihood': [], # likelihood of leanred dynamics model
        'D_KL_median': [], 'D_KL_mean': [],
        'q1_loss': [], 'policy_loss': [], 'alpha_loss': [], 'alpha': [],
        'ELBO': [],
        'step': [], 'step_reward': [],
        'test_episode': [], 'test_reward': [],
    }

    # Set up environment
    print("----------------------------------------\nTrain in {}\n----------------------------------------".format(conf.environment))
    env = gym.make(conf.environment)

    # Training set up
    agent = SAC(env.observation_space, env.action_space, device, **conf.agent)
    memory = ReplayBuffer(conf.replay_buffer_capacity, env.observation_space.shape, env.action_space.shape)
    vime = VIME(env.observation_space.shape[0], env.action_space.shape[0], device, **conf.vime) if use_vime else None
    # Load checkpoint if specified in config
    if conf.checkpoint != '':
        ckpt = torch.load(conf.checkpoint, map_location=device)
        metrics = ckpt['metrics']
        agent.load_state_dict(ckpt['agent'])
        memory.load_state_dict(ckpt['memory'])
        if use_vime:
            vime.load_state_dict(ckpt['vime'])

    def save_checkpoint():
        # Save checkpoint
        ckpt = {'metrics': metrics, 'agent': agent.state_dict(), 'memory': memory.state_dict()}
        if use_vime:
            ckpt['vime'] = vime.state_dict()
        path = os.path.join(ckpt_dir, 'checkpoint.pth')
        torch.save(ckpt, path)

        # Save agent model only
        model_ckpt = {'agent': agent.state_dict()}
        model_path = os.path.join(ckpt_dir, 'model.pth')
        torch.save(model_ckpt, model_path)

        # Save metrics only
        metrics_ckpt = {'metrics': metrics}
        metrics_path = os.path.join(ckpt_dir, 'metrics.pth')
        torch.save(metrics_ckpt, metrics_path)

    # Train agent
    init_episode = 0 if len(metrics['episode']) == 0 else metrics['episode'][-1] + 1
    pbar = tqdm.tqdm(range(init_episode, conf.episodes))
    reward_moving_avg = None
    moving_avg_coef = 0.1
    agent_update_count = 0
    total_steps = 0

    for episode in pbar:
        o = env.reset()
        rewards, curiosity_rewards = [], []
        info_gains = []
        log_likelihoods = []
        q1_losses, q2_losses, policy_losses, alpha_losses, alphas = [],[],[],[],[]

        for t in range(conf.horizon):
            if len(memory) < conf.random_sample_num:
                a = env.action_space.sample()
            else:
                a = agent.select_action(o, eval=False)

            o_next, r, done, _ = env.step(a)
            total_steps += 1
            metrics['step'].append(total_steps)
            metrics['step_reward'].append(r)
            done = False if t == env._max_episode_steps - 1 else bool(done)  # done should be False if an episode is terminated forcefully
            rewards.append(r)

            if use_vime and len(memory) >= conf.random_sample_num:
                # Calculate curiosity reward in VIME
                info_gain, log_likelihood = vime.calc_info_gain(o, a, o_next)
                assert not np.isnan(info_gain).any() and not np.isinf(info_gain).any(), "invalid information gain, {}".format(info_gains)
                info_gains.append(info_gain)
                log_likelihoods.append(log_likelihood)
                vime.memorize_episodic_info_gains(info_gain)            
                r = vime.calc_curiosity_reward(r, info_gain)
            curiosity_rewards.append(r)

            memory.append(o, a, r, o_next, done)
            o = o_next

            # Update agent
            if len(memory) >= conf.random_sample_num:
                for _ in range(conf.agent_update_per_step):
                    batch_data = memory.sample(conf.agent_update_batch_size)
                    q1_loss, q2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(batch_data, agent_update_count)
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                    policy_losses.append(policy_loss)
                    alpha_losses.append(alpha_loss)
                    alphas.append(alpha)
                    agent_update_count += 1

            if done:
                break

        if len(log_likelihoods) == 0:
            log_likelihoods.append(-np.inf)

        # Display performance
        episodic_reward = np.sum(rewards)
        reward_moving_avg = episodic_reward if reward_moving_avg is None else (1-moving_avg_coef) * reward_moving_avg + moving_avg_coef * episodic_reward
        if use_vime:
            pbar.set_description("EPISODE {}, TOTAL STEPS {}, SAMPLES {} --- Steps {}, Curiosity {:.1f}, Rwd {:.1f} (m.avg {:.1f}), Likelihood {:.2E}".format(
                episode, memory.step, len(memory), len(rewards), np.sum(curiosity_rewards), episodic_reward, reward_moving_avg, np.mean(np.exp(log_likelihoods))))
        else:
            pbar.set_description("EPISODE {}, TOTAL STEPS {}, SAMPLES {} --- Steps {}, Rwd {:.1f} (mov avg {:.1f})".format(
                episode, memory.step, len(memory), len(rewards), episodic_reward, reward_moving_avg))

        # Save episodic metrics
        metrics['episode'].append(episode)
        metrics['collected_samples'].append(total_steps)
        metrics['reward'].append(episodic_reward)
        metrics['curiosity_reward'].append(np.sum(curiosity_rewards))
        metrics['likelihood'].append(np.mean(np.exp(log_likelihoods)))
        if episode % visualize_interval == 0:
            lineplot(metrics['step'][-len(metrics['step_reward']):], metrics['step_reward'], 'stepwise_reward', log_dir, xaxis='total step')
            lineplot(metrics['episode'][-len(metrics['reward']):], metrics['reward'], 'reward', log_dir)
            lineplot(metrics['collected_samples'][-len(metrics['reward']):], metrics['reward'], 'sample-reward', log_dir, xaxis='total step')
            lineplot(metrics['episode'][-len(metrics['curiosity_reward']):], metrics['curiosity_reward'], 'curiosity_reward', log_dir)
            lineplot(metrics['episode'][-len(metrics['likelihood']):], metrics['likelihood'], 'likelihood', log_dir)
        # Agent update related metrics
        if len(policy_losses) > 0:
            metrics['q1_loss'].append(np.mean(q1_losses))
            metrics['policy_loss'].append(np.mean(policy_losses))
            metrics['alpha_loss'].append(np.mean(alpha_losses))
            metrics['alpha'].append(np.mean(alphas))
            if episode % visualize_interval == 0:
                lineplot(metrics['episode'][-len(metrics['q1_loss']):], metrics['q1_loss'], 'q1_loss', log_dir)
                lineplot(metrics['episode'][-len(metrics['policy_loss']):], metrics['policy_loss'], 'policy_loss', log_dir)
                lineplot(metrics['episode'][-len(metrics['alpha_loss']):], metrics['alpha_loss'], 'alpha_loss', log_dir)
                lineplot(metrics['episode'][-len(metrics['alpha']):], metrics['alpha'], 'alpha', log_dir)

        # Update VIME
        if use_vime and len(memory) >= conf.random_sample_num:
            for _ in range(conf.vime_update_per_episode):
                batch_s, batch_a, _, batch_s_next, _ = memory.sample(conf.vime_update_batch_size)
                elbo = vime.update_posterior(batch_s, batch_a, batch_s_next)
            metrics['ELBO'].append(elbo)
            metrics['D_KL_median'].append(np.median(info_gains))
            metrics['D_KL_mean'].append(np.mean(info_gains))
            lineplot(metrics['episode'][-len(metrics['ELBO']):], metrics['ELBO'], 'ELBO', log_dir)
            multiple_lineplot(metrics['episode'][-len(metrics['D_KL_median']):], np.array([metrics['D_KL_median'], metrics['D_KL_mean']]).T, 'D_KL', ['median', 'mean'], log_dir)

        # Test current policy
        if episode % conf.test_interval == 0:
            rewards = []
            for _ in range(conf.test_times):
                o = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    a = agent.select_action(o, eval=True)
                    o_next, r, done, _ = env.step(a)
                    episode_reward += r
                    o = o_next

                rewards.append(episode_reward)

            mean, std = np.mean(rewards), np.std(rewards)
            print("\nTEST AT EPISODE {} ({} episodes) --- Avg. Reward {:.2f} (+- {:.2f})".format(episode, conf.test_times, mean, std))

            metrics['test_episode'].append(episode)
            metrics['test_reward'].append(rewards)
            lineplot(metrics['test_episode'][-len(metrics['test_reward']):], metrics['test_reward'], 'test_reward', log_dir)
            

        # Save checkpoint
        if episode % conf.checkpoint_interval == 0:
            save_checkpoint()

    save_checkpoint()
    # Save the final model
    torch.save({'agent': agent.state_dict()}, os.path.join(ckpt_dir, 'final_model.pth'))
    



def evaluate(config_file_path: str, model_filepath: str, render: bool):
    conf_d = toml.load(open(config_file_path))
    conf = namedtuple('Config', conf_d.keys())(*conf_d.values())

    # Set random variable
    np.random.seed(int(time.time()))
    torch.manual_seed(int(time.time()))
    device = torch.device('cpu')

    env = gym.make(conf.environment)
    agent = SAC(env.observation_space, env.action_space, device, **conf.agent)
    ckpt = torch.load(model_filepath, map_location='cpu')
    agent.load_state_dict(ckpt['agent'])

    o = env.reset()
    if render:
        env.render()
    done = False
    episode_reward = 0
    t = 0
    while not done:
        a = agent.select_action(o, eval=True)
        o_next, r, done, _ = env.step(a)
        episode_reward += r
        o = o_next
        t += 1

    print("STEPS: {}, REWARD: {}".format(t, episode_reward))
    input("OK? >")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trying a walker.')
    parser.add_argument('--config', default='default_config.toml', help='Config file path')
    parser.add_argument('--save-dir', default=os.path.join('results', 'test'), help='Save directory')
    parser.add_argument('--vime', action='store_true', help='Whether to use VIME.')
    parser.add_argument('--visualize-interval', default=25, type=int, help='Interval to draw graphs of metrics.')
    parser.add_argument('--device', default='cpu', choices={'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'}, help='Device for computation.')
    parser.add_argument('-e', '--eval', action='store_true', help='Run model evaluation.')
    parser.add_argument('-m', '--model-filepath', default='', help='Path to trained model for evaluation.')
    parser.add_argument('-r', '--render', action='store_true', help='Render agent behavior during evaluation.')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.config, args.model_filepath, args.render)
    else:
        train(args.config, args.save_dir, args.vime, args.device, args.visualize_interval)
