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
<<<<<<< HEAD
        '_params_mu', '_params_rho',
=======
        '_params',
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85
        '_optim',
    ]

    def __init__(self, observation_size, action_size, eta=0.1, lamb=0.01, batch_size=10, update_iterations=500, learning_rate=0.0001, hidden_size=64, D_KL_smooth_length=10):
        super().__init__()

        self._update_iterations = update_iterations
        self._batch_size = batch_size
        self._eta = eta
        self._lamb = lamb

        self._D_KL_smooth_length = D_KL_smooth_length
<<<<<<< HEAD
        self._prev_D_KL_medians = deque(maxlen=D_KL_smooth_length)
=======
        self._prev_D_KL_medians = Queue(maxsize=D_KL_smooth_length)
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85

        self._dynamics_model = BNN(observation_size, action_size, hidden_size)
        self._params_mu = nn.Parameter(torch.zeros(self._dynamics_model.network_parameter_number))
        self._params_rho = nn.Parameter(torch.zeros(self._dynamics_model.network_parameter_number))
        self._dynamics_model.set_params(self._params_mu, self._params_rho)
        self._optim = Adam([self._params_mu, self._params_rho], lr=learning_rate) 

    def calc_curiosity_reward(self, rewards: np.ndarray, info_gains: np.ndarray):
        if len(self._prev_D_KL_medians) == 0:
            relative_gains = info_gains
        else:
<<<<<<< HEAD
            relative_gains = info_gains / np.mean(self._prev_D_KL_medians)
=======
            relative_gains = info_gains / np.mean(self._prev_D_KL_medians.queue)
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85
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
<<<<<<< HEAD
        self._dynamics_model.set_params(self._params_mu, self._params_rho)        
        ll = self._dynamics_model.calc_log_likelihood(
            torch.tensor(s_next, dtype=torch.float32).unsqueeze(0),
            torch.tensor(s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(a, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        self._optim.zero_grad()
        (-ll).backward()  # Maximize ELBO
        nabla = torch.cat([self._params_mu.grad.data, self._params_rho.grad.data])
        assert not np.isnan(nabla).any(), "ll {}\nnabla {}".format(ll, nabla)
        H = self._calc_hessian()
        assert not np.isnan(H).any(), H
=======
        l = self.calc_log_likelihood(torch.tensor(s, dtype=torch.float32).unsqueeze(0), torch.tensor(a, dtype=torch.float32).unsqueeze(0), torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        (-l).backward()  # Maximize ELBO
        nabla = torch.cat([self._params_mu.grad, self._params_mu.grad])
        H = self._calc_hessian(self)
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85

        # \frac{\lambda^2}{2} (\nabla_\phi l)^{\rm T} H^{-1} (\nabla_\phi^{\rm T} l)
        with torch.no_grad():
            return self._lamb / 2 * (nabla.pow(2) * H.pow(-1)).sum()

    def _calc_hessian(self):
        """Return diagonal elements of H = [ \frac{\partial^2 l_{D_{KL}}}{{\partial \phi_j}^2} ]_j

        \frac{\partial^2 l_{D_{KL}}}{{\partial \mu_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})}
        \frac{\partial^2 l_{D_{KL}}}{{\partial \rho_j}^2} = - \frac{1}{\log^2 (1 + e^{\phi_j})} \frac{2 e^{2 \rho_j}}{(1 + e^{rho_j})^2}
        """
<<<<<<< HEAD
=======
        params_num = len(self._params_mu)
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85
        with torch.no_grad():
            H_mu = - torch.log(1 + self._params_rho.exp()).pow(-2)
            H_rho = - 2 * (2 * self._params_rho).exp() / torch.log(1 + self._params_rho.exp()).pow(2) / (1 + self._params_rho.exp()).pow(2)
        return torch.cat([H_mu, H_rho])

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
            div_kl = self._calc_div_kl(prev_mu, prev_var)

            elbo = log_likelihood - div_kl
<<<<<<< HEAD
            self._optim.zero_grad()
=======
            self._optime.zero_grad()
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85
            (-elbo).backward()
            self._optim.step()

            # Check parameters
            assert not torch.isnan(self._params_mu).any(), self._params_mu
            assert not torch.isnan(self._params_rho).any(), self._params_rho

            # update self._params
            self._dynamics_model.set_params(self._params_mu, self._params_rho)

        return elbo.item()

    def _calc_div_kl(self, prev_mu, prev_var):
        """Calculate D_{KL} [ q(\cdot | \phi) || q(\cdot | \phi_n) ]
        = \frac{1}{2} \sum^d_i [ \log(var_{ni}) - \log(var_i) + \frac{var_i}{var_{ni}} + \frac{(\mu_i - \mu_{ni})^2}{var_{ni}} ] - \frac{d}{2}
        """
<<<<<<< HEAD
        var = (1 + self._params_rho.exp()).log().pow(2)
        return .5 * (prev_var.log() - var.log() + var / prev_var + (self._params_mu - prev_mu).pow(2) / var.data).sum() - .5 * len(self._params_mu)
=======
        vars = (1 + self._parms_rho.exp()).log().pow(2)
        return .5 * (vars.data.log() - vars.log() + vars / vars.data + (self._params_mu - self._params_mu.data).pow(2) / vars.data).sum() - .5 * len(self._params_mu)
>>>>>>> bf9808f83f1b8bac9abf119a86a7703a20778d85

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
        X = torch.cat([s, a], dim=1)
        batch_size = X.size(0)
        assert not np.isnan(s).any(), s
        assert not np.isnan(a).any(), a

        assert X.size() == torch.Size([batch_size, self._input_size]), X.size()
        X = F.relu(self._linear(self._W1_mu, self._b1_mu, self._W1_var, self._b1_var, X))
        assert not torch.isnan(X).any(), X
        X = self._linear(self._W2_mu, self._b2_mu, self._W2_var, self._b2_var, X)
        assert not torch.isnan(X).any(), X
        mean, logvar = X[:, :self._observation_size], X[:, self._observation_size:]
        return mean, logvar

    @staticmethod
    def _linear(W_mu, b_mu, W_var, b_var, X):
        """Linear forward calculation with local reparameterization trick.
        """
        assert not torch.isnan(W_mu).any(), W_mu
        assert not torch.isnan(b_mu).any(), b_mu
        assert not torch.isnan(W_var).any(), W_var
        assert not torch.isnan(b_var).any(), b_var

        gamma = X @ W_mu + b_mu
        assert not torch.isnan(gamma).any(), gamma
        delta = X @ W_var
        assert not torch.isnan(delta).any(), delta
        zeta = Normal(torch.zeros_like(delta), torch.ones_like(delta)).sample()
        assert not torch.isnan(zeta).any(), zeta
        r = gamma + delta.pow(0.5) * zeta
        assert not torch.isnan(r).any(), r
        return r


    def calc_log_likelihood(self, batch_s_next, batch_s, batch_a):
        """Calculate log likelihoods.
        Weights are sampled per data point.
        Local reparameterization trick is used instead of direct sampling of network parameters.
        """
        s_next_mean, s_next_logvar = self.infer(batch_s, batch_a)
        assert not torch.isnan(s_next_mean).any(), s_next_mean
        assert not torch.isnan(s_next_logvar).any(), s_next_logvar

        # log p(s_next)
        # = log N(s_next | s_next_mean, exp(s_next_logvar))
        # = -\frac{1}{2} \sum^d_j [ logvar_j + (s_next_j - s_next_mean)^2 exp(- logvar_j) ]  - \frac{d}{2} \log (2\pi)
        ll = - .5 * (s_next_logvar + (batch_s_next - s_next_mean).pow(2) * (- s_next_logvar).exp()).sum(dim=1) - len(s_next_mean)/2 * np.log(2 * np.pi)
        assert not torch.isnan(s_next_logvar + (batch_s_next - s_next_mean).pow(2)).any()
        assert not torch.isnan((- s_next_logvar).exp()).any()
        assert ll.size(0) == batch_s_next.size(0)
        return ll.mean()

