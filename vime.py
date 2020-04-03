"""Implementation of VIME: Variational Information Maximizing Exploration.
https://arxiv.org/abs/1605.09674
"""
from collections import deque
import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam


class VIME(nn.Module):
    _ATTRIBUTES_TO_SAVE = [
        '_D_KL_smooth_length', '_prev_D_KL_medians',
        '_eta', '_lamb',
        '_dynamics_model',
        '_params_mu', '_params_rho',
        '_optim',
    ]

    def __init__(self, observation_size, action_size, device, eta=0.1, lamb=0.01, batch_size=10, update_iterations=500, learning_rate=0.0001, hidden_size=64, D_KL_smooth_length=10, nabla_limit=100):
        super().__init__()

        self._update_iterations = update_iterations
        self._batch_size = batch_size
        self._eta = eta
        self._lamb = lamb
        self._nabla_limit = nabla_limit
        self._device = device

        self._D_KL_smooth_length = D_KL_smooth_length
        self._prev_D_KL_medians = deque(maxlen=D_KL_smooth_length)

        self._dynamics_model = BNN(observation_size, action_size, hidden_size)
        self._params_mu = nn.Parameter(torch.zeros(self._dynamics_model.network_parameter_number))
        self._params_rho = nn.Parameter(torch.zeros(self._dynamics_model.network_parameter_number))
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
        """
        self._dynamics_model.set_params(self._params_mu, self._params_rho) # necessary to calculate new gradient
        ll = self._dynamics_model.calc_log_likelihood(
            torch.tensor(s_next, dtype=torch.float32).unsqueeze(0),
            torch.tensor(s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
        self._optim.zero_grad()
        (-ll).backward()  # Calculate gradient \nabla_\phi l ( = \nalba_\phi -E_{\theta \sim q(\cdot | \phi)}[ \log p(s_{t+1} | \s_t, a_t, \theta) ] )
        nabla = torch.cat([self._params_mu.grad.data, self._params_rho.grad.data])
        nabla = torch.clamp(nabla, min=-self._nabla_limit, max=self._nabla_limit) # limit absolute value of nabla
        assert not np.isnan(nabla).any() and not np.isinf(nabla).any(), "ll {}\nnabla {}".format(ll, nabla)
        H = self._calc_hessian()
        assert not torch.isinf(nabla.data.pow(2)).any(), "nabla_rho\n{}\nnabla_rho^2\n{}".format(nabla.data, nabla.data.pow(2))
        assert not torch.isinf(H.pow(-1)).any(), H.pow(-1)
        
        # \frac{\lambda^2}{2} (\nabla_\phi l)^{\rm T} H^{-1} (\nabla_\phi^{\rm T} l)
        with torch.no_grad():
            info_gain = self._lamb ** 2 / 2 * (nabla.pow(2) * H.pow(-1)).sum()
            assert not torch.isinf(info_gain).any(), info_gain
        return info_gain.item()

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
        assert not np.isnan(H).any() and not np.isinf(H).any(), H
        return H

    def update_posterior(self, memory):
        """
        Return
        ---
        loss: (float)
        """
        prev_mu, prev_var = self._params_mu.data, (1 + self._params_rho.data.exp()).log().pow(2)
        for i in range(self._update_iterations):
            batch_s, batch_a, _, batch_s_next, _ = memory.sample(self._batch_size)
            batch_s, batch_a, batch_s_next = torch.tensor(batch_s, dtype=torch.float32), torch.tensor(batch_a, dtype=torch.float32), torch.tensor(batch_s_next, dtype=torch.float32)
            self._dynamics_model.set_params(self._params_mu, self._params_rho)
            log_likelihood = self._dynamics_model.calc_log_likelihood(batch_s_next, batch_s, batch_a)
            assert not torch.isnan(log_likelihood).any() and not torch.isinf(log_likelihood).any(), log_likelihood.item()
            div_kl = self._calc_div_kl(prev_mu, prev_var)
            assert not torch.isnan(div_kl).any() and not torch.isinf(div_kl).any(), div_kl.item()

            elbo = log_likelihood - div_kl
            assert not torch.isnan(elbo).any() and not torch.isinf(elbo).any(), elbo.item()

            self._optim.zero_grad()
            (-elbo).backward()
            self._optim.step()

            # Check parameters
            assert not torch.isnan(self._params_mu).any() and not torch.isinf(self._params_mu).any(), self._params_mu
            assert not torch.isnan(self._params_rho).any() and not torch.isinf(self._params_rho).any(), self._params_rho

            # update self._params
            self._dynamics_model.set_params(self._params_mu, self._params_rho)

        return elbo.item()

    def _calc_div_kl(self, prev_mu, prev_var):
        """Calculate D_{KL} [ q(\cdot | \phi) || q(\cdot | \phi_n) ]
        = \frac{1}{2} \sum^d_i [ \log(var_{ni}) - \log(var_i) + \frac{var_i}{var_{ni}} + \frac{(\mu_i - \mu_{ni})^2}{var_{ni}} ] - \frac{d}{2}
        """
        var = (1 + self._params_rho.exp()).log().pow(2)
        return .5 * ( prev_var.log() - var.log() + var / prev_var + (self._params_mu - prev_mu).pow(2) / prev_var ).sum() - .5 * len(self._params_mu)

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

        

class BNN:
    def __init__(self, observation_size, action_size, hidden_size):
        self._input_size = observation_size + action_size
        self._observation_size = observation_size

        self._W1_mu = torch.zeros(observation_size + action_size, hidden_size)
        self._b1_mu = torch.zeros(hidden_size)
        self._W2_mu = torch.zeros(hidden_size, observation_size * 2)
        self._b2_mu = torch.zeros(observation_size * 2) # to obtain mean and logvar

        self._W1_var = torch.ones(observation_size + action_size, hidden_size)
        self._b1_var = torch.ones(hidden_size)
        self._W2_var = torch.ones(hidden_size, observation_size * 2)
        self._b2_var = torch.ones(observation_size * 2) # to obtain mean and logvar

        W1_size, b1_size = np.prod(self._W1_mu.size()), np.prod(self._b1_mu.size())
        W2_size, b2_size = np.prod(self._W2_mu.size()), np.prod(self._b2_mu.size())
        self._network_parameter_num = W1_size + b1_size + W2_size + b2_size

    @property
    def network_parameter_number(self):
        return self._network_parameter_num

    def set_params(self, params_mu, params_rho):
        """Set a vector of parameters into weights and biases.
        """
        assert params_mu.size() == torch.Size([self._network_parameter_num]), "expected {}, got {}".format(self._network_parameter_num, params_mu.size())
        assert params_rho.size() == torch.Size([self._network_parameter_num]), "expected {}, got {}".format(self._network_parameter_num, params_rho.size())
        W1_size, b1_size = np.prod(self._W1_mu.size()), np.prod(self._b1_mu.size())
        W2_size, b2_size = np.prod(self._W2_mu.size()), np.prod(self._b2_mu.size())

        # Set \mus
        self._W1_mu = params_mu[0 : W1_size].reshape(self._W1_mu.size())
        self._b1_mu = params_mu[W1_size : W1_size + b1_size].reshape(self._b1_mu.size())
        self._W2_mu = params_mu[W1_size + b1_size : W1_size + b1_size + W2_size].reshape(self._W2_mu.size())
        self._b2_mu = params_mu[W1_size + b1_size + W2_size : W1_size + b1_size + W2_size + b2_size].reshape(self._b2_mu.size())

        # Set \rho
        self._W1_var = torch.log(1 + params_rho[0 : W1_size].reshape(self._W1_var.size()).exp()).pow(2)
        self._b1_var = torch.log(1 + params_rho[W1_size : W1_size + b1_size].reshape(self._b1_var.size()).exp()).pow(2)
        self._W2_var = torch.log(1 + params_rho[W1_size + b1_size : W1_size + b1_size + W2_size].reshape(self._W2_var.size()).exp()).pow(2)
        self._b2_var = torch.log(1 + params_rho[W1_size + b1_size + W2_size : W1_size + b1_size + W2_size + b2_size].reshape(self._b2_var.size()).exp()).pow(2)

    def get_params(self):
        params_mu = torch.cat([self._W1_mu.data.reshape(-1), self._b1_mu.data.reshape(-1), self._W2_mu.data.reshape(-1), self._b2_mu.data.reshape(-1)])
        params_rho = torch.cat([
            (self._W1_var.data.pow(.5).exp() - 1).log().reshape(-1), (self._b1_var.data.pow(.5).exp() - 1).log().reshape(-1),
            (self._W2_var.data.pow(.5).exp() - 1).log().reshape(-1), (self._b2_var.data.pow(.5).exp() - 1).log().reshape(-1),
        ])
        return params_mu, params_rho

    def infer(self, s, a):
        """Forward calculate with local reparameterization.
        """
        assert not np.isnan(s).any() and not np.isinf(s).any(), s
        assert not np.isnan(a).any() and not np.isinf(a).any(), a

        X = torch.cat([s, a], dim=1)
        batch_size = X.size(0)
        assert X.size() == torch.Size([batch_size, self._input_size]), X.size()
        X = F.relu(self._linear(self._W1_mu, self._b1_mu, self._W1_var, self._b1_var, X))
        assert not torch.isnan(X).any() and not torch.isinf(X).any(), X
        X = self._linear(self._W2_mu, self._b2_mu, self._W2_var, self._b2_var, X)
        assert not torch.isnan(X).any() and not torch.isinf(X).any(), X
        mean, logvar = X[:, :self._observation_size], X[:, self._observation_size:]
        return mean, logvar

    @staticmethod
    def _linear(W_mu, b_mu, W_var, b_var, X):
        """Linear forward calculation with local reparameterization trick.
        """
        assert not torch.isnan(W_mu).any() and not torch.isinf(W_mu).any(), W_mu
        assert not torch.isnan(b_mu).any() and not torch.isinf(b_mu).any(), b_mu
        assert not torch.isnan(W_var).any() and not torch.isinf(W_var).any(), W_var
        assert not torch.isnan(b_var).any() and not torch.isinf(b_var).any(), b_var

        gamma = X @ W_mu + b_mu
        delta = X.pow(2) @ W_var + b_var
        assert not torch.isnan(gamma).any(), gamma
        assert not torch.isnan(delta).any(), delta
        assert not torch.isnan(delta.pow(.5)).any(), delta.pow(.5)

        zeta = Normal(torch.zeros_like(delta), torch.ones_like(delta)).sample()
        r = gamma + delta.pow(0.5) * zeta
        assert not torch.isnan(r).any() and not torch.isinf(r).any(), r
        return r

    def calc_log_likelihood(self, batch_s_next, batch_s, batch_a):
        """Calculate log likelihoods.
        Weights are sampled per data point.
        Local reparameterization trick is used instead of direct sampling of network parameters.
        """
        s_next_mean, s_next_logvar = self.infer(batch_s, batch_a)
        # print('----------')
        # print("s_{t+1} mean:", s_next_mean[0])
        # print("s_{t+1} logvar:", s_next_logvar[0])
        # print("p(s_{t+1}) =", torch.distributions.Normal(s_next_mean[0], (s_next_logvar[0] / 2).exp()).log_prob(batch_s_next[0]).sum().exp().item())
        # print('----------')
        assert not torch.isnan(s_next_mean).any() and not torch.isinf(s_next_mean).any(), s_next_mean
        assert not torch.isnan(s_next_logvar).any() and not torch.isinf(s_next_logvar).any(), s_next_logvar

        # log p(s_next)
        # = log N(s_next | s_next_mean, exp(s_next_logvar))
        # = -\frac{1}{2} \sum^d_j [ logvar_j + (s_next_j - s_next_mean)^2 exp(- logvar_j) ]  - \frac{d}{2} \log (2\pi)
        ll = - .5 * ( s_next_logvar + (batch_s_next - s_next_mean).pow(2) * (- s_next_logvar).exp() ).sum(dim=1) - .5 * self._observation_size * np.log(2 * np.pi)
        assert not torch.isinf((- s_next_logvar).exp()).any(), "s_next_logvar\n{}\nexp(-s_next_logvar)\n{}".format(s_next_logvar, (- s_next_logvar).exp())
        assert not torch.isinf((batch_s_next - s_next_mean).pow(2)).any(), (batch_s_next - s_next_mean).pow(2)
        assert not torch.isinf((batch_s_next - s_next_mean).pow(2) * (- s_next_logvar).exp()).any(), (batch_s_next - s_next_mean).pow(2) * (- s_next_logvar).exp()
        # print("p(s_{t+1} | s_t, a_t, theta) =", ll.exp().mean().item(), "\tloglikelihood =", ll.mean().item())
        assert ll.size(0) == batch_s_next.size(0)
        return ll.mean()

