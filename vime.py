"""Implementation of VIME: Variational Information Maximizing Exploration.
https://arxiv.org/abs/1605.09674
"""
from queue import Queue
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
        '_mus', '_rhos',
        '_dynamics_model', '_params', '_optim',
    ]

    def __init__(self, observation_size, action_size, eta=0.1, lamb=0.01, batch_size=10, update_iterations=500, learning_rate=0.0001, hidden_size=64, D_KL_smooth_length=10):
        self._update_iterations = update_iterations
        self._batch_size = batch_size
        self._eta = eta
        self._lamb = lamb

        self._mus = torch.zeros()

        self._D_KL_smooth_length = D_KL_smooth_length
        self._prev_D_KL_medians = Queue(maxsize=D_KL_smooth_length)


        self._dynamics_model = BNN(observation_size, action_size)
        self._params = nn.Parameter(torch.zeros(self._dynamics_model.parameter_number))
        self._dynamics_model.set_params(self._params)
        self._optim = Adam(self._params, lr=learning_rate) 


    def calc_curiosity_reward(self, rewards: np.ndarray, info_gains: np.ndarray):
        if self._prev_D_KL_medians.empty():
            relative_gains = info_gains
        else:
            relative_gains = info_gains / np.mean(self._prev_D_KL_medians.queue)
        return rewards + self._eta * relative_gain

    def memorize_episodic_info_gains(self, info_gains: np.array):
        """Call this method after collecting a trajectory to save a median of infomation gains throughout the episode.
        Params
        ---
        info_gains: array of D_KLs throughout an episode
        """
        self._prev_D_KL_medians.put(np.median(info_gains))

    def calc_info_gain(self, s, a, s_next):
        """Calculate information gain D_KL[ q( /cdot | \phi') || q( /cdot | \phi_n) ].
        """
        l = self.calc_log_likelihood(s.unsqueeze(0), a.unsqueeze(0), s_next.unsqueeze(0)).squeeze(0)
        torch.backward(-l)  # Maximize ELBO
        nabla = self._params.grad
        H = self._calc_hessian(self)

        # \frac{\lambda^2}{2} (\nabla_\phi l)^{\rm T} H^{-1} (\nabla_\phi^{\rm T} l)
        return self._lamb / 2 * (nabla.pow(2) * H.pow(-1)).sum()

    def _calc_hessian(self):
        """Return diagonal elements of H = [ \frac{\partial^2 l_{D_{KL}}}{{\partial \phi_j}^2} ]_j

        \frac{\partial^2 l_{D_{KL}}}{{\partial \mu_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})}
        \frac{\partial^2 l_{D_{KL}}}{{\partial \rho_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})} \frac{2 e^{2 \rho_j}}{(1 + e^{rho_j})^2}
        """
        params_num = len(self._params) / 2
        params_rhos = self._params[params_num:]
        H_mu = - torch.log(1 + params_rhos.exp()).pow(-2)
        H_rho = - 2 * (2 * params_rhos).exp() / torch.log(1 + params_rhos.exp()).pow(2) / (1 + params_rhos.exp()).pow(2)
        return torch.cat([H_mu, H_rho])

    def calc_log_likelihood(self, batch_s, batch_a, batch_s_next):
        """Calculate E[ \log p(s_{t+1} | s_t, a_t; \phi) ].
        """
        log_likelihood = self._dynamics_model.calc_log_likelihood(batch_s_next, batch_s, batch_a) 
        return log_likelihood


    def update_posterior(self, memory):
        """
        Return
        ---
        loss: (float)
        """
        loss = 0.
        for i in range(self._update_iterations):
            batch_s, batch_a, _, batch_s_next, _ = memory.sample(self.batch_size)
            log_likelihood = self.calc_log_likelihood(batch_s, batch_a, _next)
            div_kl = self._calc_div_kl()

            elbo = log_likelihood - div_kl
            self._optime.zero_grad()
            torch.backward(-elbo)
            self._optim.step()

            # update self._params
            self._dynamics_model.set_params(self._params)

        return -elbo

    def _calc_div_kl(self):
        """Calculate D_{KL} [ q(\cdot | \phi) || q(\cdot | \phi_n) ]
        = \frac{1}{2} \sum^d_i [ \log(var_{ni}) - \log(var_i) + \frac{var_i}{var_{ni}} + \frac{(\mu_i - \mu_{ni})^2}{var_{ni}} ] - \frac{d}{2}
        """
        rhos = self._params[len(self._params) // 2:]
        vars = (1 + rhos.exp()).log().pow(2)
        current_vars = vars.data
        mus = self._params[:len(self._params) // 2]
        current_mus = mus.data
        return .5 * (current_vars.log() - vars.log() + vars / current_vars + (mus - current_mus).pow(2) / current_vars).sum() - .5 * len(mus)

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
        self._W2_mu = torch.zeros(hidden_size, observation_size)
        self._b2_mu = torch.zeros(observation_size * 2) # to obtain mean and logvar

        self._W1_var = torch.ones(observation_size + action_size, hidden_size)
        self._b1_var = torch.ones(hidden_size)
        self._W2_var = torch.ones(hidden_size, observation_size)
        self._b2_var = torch.ones(observation_size * 2) # to obtain mean and logvar

        self._parameter_num = (np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size()) + np.prod(self._W2_mu.size()) + np.prod(self._b2_mu.size())) * 2  # \mu and \sigma^2

    @property
    def parameter_number(self):
        return self._parameter_num

    def set_params(self, params):
        """Set a vector of parameters into weights and biases.
        """
        assert params.size() == torch.Size([self._parameter_num]), "expected {}, got {}".format(self._parameter_num, params.size())
        # Set \mus
        params_mu = params[:self._parameter_num//2]
        self._W1_mu = params_mu[0:np.prod(self._W1_mu.size())].reshape(self._W1_mu.size())
        self._b1_mu = params_mu[np.prod(self._W1_mu.size()):np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size())].reshape(self._b1_mu.size())
        self._W2_mu = params_mu[np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size()):np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size()) + np.prod(self._W2_mu.size())].reshape(self._W1_mu.size())
        self._b2_mu = params_mu[np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size()) + np.prod(self._W2_mu.size()):np.prod(self._W1_mu.size()) + np.prod(self._b1_mu.size()) + np.prod(self._W2_mu.size()) + np.prod(self._b2_mu.size())].reshape(self._b1_mu.size())

        # Set \rho
        params_rho = params[self._parameter_num//2:]
        self._W1_var = torch.log(1 + params_rho[0:np.prod(self._W1_var.size())].reshape(self._W1_var.size()).exp()).pow(2)
        self._b1_var = torch.log(1 + params_rho[np.prod(self._W1_var.size()):np.prod(self._W1_var.size()) + np.prod(self._b1_var.size())].reshape(self._b1_var.size()).exp()).pow(2)
        self._W2_var = torch.log(1 + params_rho[np.prod(self._W1_var.size()) + np.prod(self._b1_var.size()):np.prod(self._W1_var.size()) + np.prod(self._b1_var.size()) + np.prod(self._W2_var.size())].reshape(self._W1_var.size()).exp()).pow(2)
        self._b2_var = torch.log(1 + params_rho[np.prod(self._W1_var.size()) + np.prod(self._b1_var.size()) + np.prod(self._W2_var.size()):np.prod(self._W1_var.size()) + np.prod(self._b1_var.size()) + np.prod(self._W2_var.size()) + np.prod(self._b2_var.size())].reshape(self._b1_var.size()).exp()).pow(2)

    def get_params(self):
        return torch.cat([
            self._W1_mu.data.reshape(-1), self._b1_mu.data.reshape(-1), self._W2_mu.data.reshape(-1), self._b2_mu.data.reshape(-1),
            (self._W1_var.data.pow(.5).exp() - 1).log().reshape(-1), (self._b1_var.data.pow(.5).exp() - 1).log().reshape(-1),
            (self._W2_var.data.pow(.5).exp() - 1).log().reshape(-1), (self._b2_var.data.pow(.5).exp() - 1).log().reshape(-1),
        ])

    def infer(self, s, a):
        """Forward calculate with local reparameterization.
        """
        X = torch.cat([s, a], dim=1)
        batch_size = X.size(0)
        assert X.size() == torch.Size(batch_size, self._input_size), X.size()
        X = F.relu(self._linear(self._W1_mu, self._b1_mu, self._W1_var, self._b1_var, X))
        X = self._linear(self._W1_mu, self._b1_mu, self._W1_var, self._b1_var, X)
        mean, logvar = X[:, :self._observation_size], X[:, self._observation_size:]
        return mean, logvar

    @staticmethod
    def _linear(W_mu, b_mu, W_var, b_var, X):
        """Linear forward calculation with local reparameterization trick.
        """
        gamma = X @ W_mu + b_mu
        delta = X @ W_var
        zeta = Normal(torch.zeros_like(delta), torch.ones_like(delta))
        return  gamma + delta.pow(0.5) * zeta


    def calc_log_likelihood(self, batch_s_next, batch_s, batch_a):
        """Calculate log likelihoods.
        Weights are sampled per data point.
        Local reparameterization trick is used instead of direct sampling of network parameters.
        """
        s_next_mean, s_next_logvar = self.infer(batch_s, batch_a)

        # log p(s_next)
        # = log N(s_next | s_next_mean, exp(s_next_logvar))
        # = -\frac{1}{2} \sum^d_j [ logvar_j + (s_next_j - s_next_mean)^2 exp(- logvar_j) ]  - \frac{d}{2} \log (2\pi)
        l = - .5 (s_next_logvar + (batch_s_next - s_next_mean).pow(2) * exp(- s_next_logvar)).sum(dim=1) - d/2 * np.log(2 * np.pi)
        assert l.size(0) == batch_s_next.size(0)
        return l.mean()

