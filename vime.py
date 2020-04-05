"""Implementation of VIME: Variational Information Maximizing Exploration.
https://arxiv.org/abs/1605.09674
"""
from collections import deque
import numpy as np
import torch
from torch import nn 
from torch.optim import Adam

from bayesian_neural_network import BNN


class VIME(nn.Module):
    _ATTRIBUTES_TO_SAVE = [
        '_D_KL_smooth_length', '_prev_D_KL_medians',
        '_eta', '_lamb',
        '_dynamics_model',
        '_params_mu', '_params_rho',
        '_H',
        '_optim',
    ]

    def __init__(self, observation_size, action_size, device='cpu', eta=0.1, lamb=0.01, batch_size=10, update_iterations=500,
            learning_rate=0.0001, hidden_layers=2, hidden_layer_size=64, D_KL_smooth_length=10, max_logvar=2., min_logvar=-10.):
        super().__init__()

        self._update_iterations = update_iterations
        self._batch_size = batch_size
        self._eta = eta
        self._lamb = lamb
        self._device = device

        self._D_KL_smooth_length = D_KL_smooth_length
        self._prev_D_KL_medians = deque(maxlen=D_KL_smooth_length)

        self._dynamics_model = BNN(observation_size + action_size, observation_size, hidden_layers, hidden_layer_size, max_logvar, min_logvar)
        init_params_mu, init_params_rho = self._dynamics_model.get_parameters()
        self._params_mu = nn.Parameter(init_params_mu.to(device))
        self._params_rho = nn.Parameter(init_params_rho.to(device))
        self._H = self._calc_hessian()
        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        self._optim = Adam([self._params_mu, self._params_rho], lr=learning_rate) 

    def calc_curiosity_reward(self, rewards: np.ndarray, info_gains: np.ndarray):
        if len(self._prev_D_KL_medians) == 0:
            relative_gains = info_gains
        else:
            relative_gains = info_gains / np.mean(self._prev_D_KL_medians)
        return rewards + self._eta * relative_gains

    def memorize_episodic_info_gains(self, info_gains: np.array):
        """Call this method after collecting a trajectory to save a median of infomation gains throughout the episode.
        Params
        ---
        info_gains: array of D_KLs throughout an episode
        """
        self._prev_D_KL_medians.append(np.median(info_gains))

    def calc_info_gain(self, s, a, s_next):
        """Calculate information gain D_KL[ q( /cdot | \phi') || q( /cdot | \phi_n) ].

        Return info_gain, log-likelihood of each sample \log p(s_{t+1}, a_t, s_)
        """
        self._dynamics_model.set_params(self._params_mu, self._params_rho) # necessary to calculate new gradient
        ll = self._dynamics_model.log_likelihood(
            torch.tensor(np.concatenate([s, a]), dtype=torch.float32).unsqueeze(0).to(self._device),
            torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(self._device))
        l = - ll.mean()

        self._optim.zero_grad()
        l.backward()  # Calculate gradient \nabla_\phi l ( = \nalba_\phi -E_{\theta \sim q(\cdot | \phi)}[ \log p(s_{t+1} | \s_t, a_t, \theta) ] )
        nabla = torch.cat([self._params_mu.grad.data, self._params_rho.grad.data])

        # \frac{\lambda^2}{2} (\nabla_\phi l)^{\rm T} H^{-1} (\nabla_\phi^{\rm T} l)
        with torch.no_grad():
            info_gain = .5 * self._lamb ** 2 * torch.sum(nabla.pow(2) * self._H.pow(-1))
        return info_gain.cpu().item(), ll.mean().detach().cpu().item()

    def _calc_hessian(self):
        """Return diagonal elements of H = [ \frac{\partial^2 l_{D_{KL}}}{{\partial \phi_j}^2} ]_j

        \frac{\partial^2 l_{D_{KL}}}{{\partial \mu_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})}
        \frac{\partial^2 l_{D_{KL}}}{{\partial \rho_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})} \frac{2 e^{2 \rho_j}}{(1 + e^{rho_j})^2}
        """
        with torch.no_grad():
            denomi = 1 + self._params_rho.exp()
            log_denomi = denomi.log()
            H_mu = log_denomi.pow(-2)
            H_rho = 2 * torch.exp(2 * self._params_rho) / (denomi * log_denomi).pow(2)
            H = torch.cat([H_mu, H_rho])
        return H

    def _calc_div_kl(self, prev_mu, prev_var):
        """Calculate D_{KL} [ q(\cdot | \phi) || q(\cdot | \phi_n) ]
        = \frac{1}{2} \sum^d_i [ \log(var_{ni}) - \log(var_i) + \frac{var_i}{var_{ni}} + \frac{(\mu_i - \mu_{ni})^2}{var_{ni}} ] - \frac{d}{2}
        """
        var = (1 + self._params_rho.exp()).log().pow(2)
        return .5 * ( prev_var.log() - var.log() + var / prev_var + (self._params_mu - prev_mu).pow(2) / prev_var ).sum() - .5 * len(self._params_mu)

    def update_posterior(self, memory):
        """
        Return
        ---
        loss: (float)
        """
        prev_mu, prev_var = self._params_mu.data, (1 + self._params_rho.data.exp()).log().pow(2)
        for i in range(self._update_iterations):
            batch_s, batch_a, _, batch_s_next, _ = memory.sample(self._batch_size)
            batch_s = torch.tensor(batch_s, dtype=torch.float32).to(self._device)
            batch_a = torch.tensor(batch_a, dtype=torch.float32).to(self._device)
            batch_s_next = torch.tensor(batch_s_next, dtype=torch.float32).to(self._device)

            self._dynamics_model.set_params(self._params_mu, self._params_rho)
            log_likelihood = self._dynamics_model.log_likelihood(torch.cat([batch_s, batch_a], dim=1), batch_s_next).mean()
            div_kl = self._calc_div_kl(prev_mu, prev_var)

            elbo = log_likelihood - div_kl
            assert not torch.isnan(elbo).any() and not torch.isinf(elbo).any(), elbo.item()

            self._optim.zero_grad()
            (-elbo).backward()
            self._optim.step()

            # Update hessian
            self._H = self._calc_hessian()

            # Check parameters
            assert not torch.isnan(self._params_mu).any() and not torch.isinf(self._params_mu).any(), self._params_mu
            assert not torch.isnan(self._params_rho).any() and not torch.isinf(self._params_rho).any(), self._params_rho

            # update self._params
            self._dynamics_model.set_params(self._params_mu, self._params_rho)

        return elbo.item()

    def state_dict(self):
        return {
            k: getattr(self, k).state_dict() if hasattr(getattr(self, k), 'state_dict') else getattr(self, k)
            for k in self._ATTRIBUTES_TO_SAVE
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if hasattr(getattr(self, k), 'load_state_dict'):
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)         

        
