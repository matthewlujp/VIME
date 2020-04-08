import numpy as np
import torch
from torch import nn 
from torch.nn import functional as F
from torch.distributions.normal import Normal


def _elements(t: torch.Tensor):
    return np.prod(t.shape)


class _BayesianLinerLayer(nn.Module):
    """A linear layer which samples network parameters on forward calculation. 
    Local re-parameterization trick is used instead of direct sampling of network parameters.
    """
    def __init__(self, fan_in: int, fan_out: int):
        super().__init__()
        self._fan_in, self._fan_out = fan_in, fan_out

        self._W_mu = torch.normal(torch.zeros(fan_in, fan_out), torch.ones(fan_in, fan_out)) # N(0, 1)
        self._W_rho = torch.log(torch.exp(torch.ones(fan_in, fan_out) * 0.5) - 1.) # log(e^0.5 - 1) to make \sigma_0 = 0.5
        self._b_mu = torch.normal(torch.zeros(fan_out), torch.ones(fan_out)) # N(0, 1)
        self._b_rho = torch.log(np.exp(torch.ones(fan_out) * .5) - 1.) # log(e^0.5 - 1) to make \sigma_0 = 0.5

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(self._b_rho)
        self._parameter_number = _elements(self._W_mu) + _elements(self._b_mu)
        self._distributional_parameter_number = _elements(self._W_mu) + _elements(self._W_rho) + _elements(self._b_mu) + _elements(self._b_rho)

    @staticmethod
    def _rho2var(rho):
        return torch.log(1. + torch.exp(rho)).pow(2)
        
    @property
    def parameter_number(self):
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return all parameters in this layer as vectors of mu and rho.
        """
        params_mu = torch.cat([self._W_mu.data.reshape(-1), self._b_mu.data.reshape(-1)])
        params_rho = torch.cat([self._W_rho.data.reshape(-1), self._b_rho.data.reshape(-1)])
        return params_mu, params_rho

    def set_parameters(self, params_mu: torch.Tensor, params_rho: torch.Tensor):
        """Receive parameters (mu and rho) as vectors and set them.
        """
        assert params_mu.size() == torch.Size([self._parameter_number])
        assert params_rho.size() == torch.Size([self._parameter_number])

        self._W_mu = params_mu[: _elements(self._W_mu)].reshape(self._W_mu.size())
        self._b_mu = params_mu[_elements(self._W_mu) :].reshape(self._b_mu.size())

        self._W_rho = params_rho[: _elements(self._W_rho)].reshape(self._W_rho.size())
        self._b_rho = params_rho[_elements(self._W_rho) :].reshape(self._b_rho.size())

        self._W_var, self._b_var = self._rho2var(self._W_rho), self._rho2var(self._b_rho)

    def forward(self, X, share_paremeters_among_samples=True):
        """Linear forward calculation with local re-parameterization trick.
        params
        ---
        X: (batch, input_size)
        share_paremeters_among_samples: (bool) Use the same set of parameters for samples in a batch

        return
        ---
        r: (batch, output_size)
        """
        gamma = X @ self._W_mu + self._b_mu
        delta = X.pow(2) @ self._W_var + self._b_var

        if share_paremeters_among_samples:
            zeta = Normal(torch.zeros(1, self._fan_out), torch.ones(1, self._fan_out)).sample().repeat([X.size(0), 1])
        else:
            zeta = Normal(torch.zeros(X.size(0), self._fan_out), torch.ones(X.size(0), self._fan_out)).sample()
        zeta = zeta.to(X.device)
        r = gamma + delta.pow(0.5) * zeta
        return r


class BNN:
    def __init__(self, input_size, output_size, hidden_layers, hidden_layer_size, max_logvar, min_logvar):
        self._input_size = input_size
        self._output_size = output_size
        self._max_logvar = max_logvar
        self._min_logvar = min_logvar

        self._hidden_layers = []
        fan_in = self._input_size
        self._parameter_number = 0
        for _ in range(hidden_layers):
            l = _BayesianLinerLayer(fan_in, hidden_layer_size)
            self._hidden_layers.append(l)   
            self._parameter_number += l.parameter_number
            fan_in = hidden_layer_size
        self._out_layer = _BayesianLinerLayer(fan_in, output_size * 2)
        self._parameter_number += self._out_layer.parameter_number
        self._distributional_parameter_number = self._parameter_number * 2

    @property
    def network_parameter_number(self):
        """The number elements in theta."""
        return self._parameter_number

    @property
    def distributional_parameter_number(self):
        """The number elements in phi."""
        return self._distributional_parameter_number

    def get_parameters(self):
        """Return mu and rho as a tuple of vectors. 
        """
        params_mu, params_rho = zip(*[l.get_parameters() for l in self._hidden_layers + [self._out_layer]])
        return torch.cat(params_mu), torch.cat(params_rho)
        
    def set_params(self, params_mu, params_rho):
        """Set a vector of parameters into weights and biases.
        """
        assert params_mu.size() == torch.Size([self._parameter_number]), "expected a vector of {}, got {}".format(self._parameter_number, params_mu.size())
        assert params_rho.size() == torch.Size([self._parameter_number]), "expected a vector of {}, got {}".format(self._parameter_number, params_rho.size())

        begin = 0
        for l in self._hidden_layers + [self._out_layer]:
            end = begin + l.parameter_number
            l.set_parameters(params_mu[begin : end], params_rho[begin : end])
            begin = end

    def infer(self, X, share_paremeters_among_samples=True):
        for layer in self._hidden_layers:
            X = F.relu(layer(X, share_paremeters_among_samples))
        X = self._out_layer(X, share_paremeters_among_samples)
        mean, logvar = X[:, :self._output_size], X[:, self._output_size:]
        logvar = torch.clamp(logvar, min=self._min_logvar, max=self._max_logvar)
        return mean, logvar

    def log_likelihood(self, input_batch, output_batch):
        """Calculate an expectation of log likelihood.
        Mote Carlo approximation using a single parameter sample,
        i.e., E_{theta ~ q(* | phi)} [ log p(D | theta)] ~ log p(D | theta_1)
        """
        output_mean, output_logvar = self.infer(input_batch, share_paremeters_among_samples=True)

        # log p(s_next)
        # = log N(output_batch | output_mean, exp(output_logvar))
        # = -\frac{1}{2} \sum^d_j [ logvar_j + (s_next_j - output_mean)^2 exp(- logvar_j) ]  - \frac{d}{2} \log (2\pi)
        ll = - .5 * ( output_logvar + (output_batch - output_mean).pow(2) * (- output_logvar).exp() ).sum(dim=1) - .5 * self._output_size * np.log(2 * np.pi)
        return ll.mean()

