import torch
import numpy as np
import torch.nn as nn
import pdb
from scipy.stats import truncnorm

# Variable initialization functions
# Copied directly from TensorFlow code
def truncated_normal(size, stddev=1, mean=0):
    mu, sigma = mean, stddev
    lower, upper= -2 * sigma, 2 * sigma
    X = truncnorm(
        (lower - mu) / sigma, (upper + mu) / sigma, loc=mu, scale=sigma)
    X_tensor = torch.Tensor(data = X.rvs(size)).to(device = device)
    return X_tensor

class BayesianLinear(nn.Module):
    """Single Bayesian MFVI layer for MFVI NN.
    Arguments:
        dim_in: Dimension of input layer.
        dim_out: Dimension of output layer.

    Parameters:
        weight_mu: Current mean for weight parameters.
        weight_log_var: Current (log) variance for weight parameters.
        bias_mu: Current mean for bias parameters.
        bias_log_var: Current (log) variance for bias parameters.

        weight_mu_prior: Mean of prior distribution over weights.
        weight_var_prior: Variance of prior distribution over weights.
        bias_mu_prior: Mean of prior distribution over biases.
        bias_var_prior: Variance of prior distribution over biases.
    """
    def __init__(self, dim_in, dim_out, prev_means=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Initialise values for means and (log) variance
        self.weight_mu = nn.Parameter(truncated_normal([dim_in, dim_out],
                                                       stddev=0.1))
        self.weight_log_var = nn.Parameter(-6 * torch.ones([dim_in, dim_out]))

        self.bias_mu = nn.Parameter(truncated_normal([dim_out],
                                                      stddev=0.1))
        self.bias_log_var = nn.Parameter(-6 * torch.ones([dim_out]))

        # Initialise prior values for means and variance
        self.weight_mu_prior = torch.zeros([dim_in, dim_out])
        self.bias_mu_prior = torch.zeros([dim_out])

        self.weight_var_prior = torch.ones([dim_in, dim_out])
        self.bias_var_prior = torch.ones([dim_out])

    def forward(self, input):
        # Sample weight epsilon and bias epsilon. This is the reparameterisation
        # trick, and allows gradient descent to be performed.
        weight_eps = torch.normal(torch.zeros([self.dim_in, self.dim_out]),
                                  torch.ones([self.dim_in, self.dim_out]))
        bias_eps = torch.normal(torch.zeros(self.dim_out),
                                torch.ones(self.dim_out))

        # Generate weight and bias samples.
        # Note: std = exp(0.5 * log(var))
        weight = torch.add(weight_eps * torch.exp(0.5 * self.weight_log_var),
                           self.weight_mu)
        bias = torch.add(bias_eps * torch.exp(0.5 * self.bias_log_var),
                         self.bias_mu)

        # Pre-activation values (i.e. outputs from the layer)
        pre = torch.matmul(input, weight) + bias

        return pre

    def KL_loss(self):
        kl = 0

        # Equal to the number of dimensions (i.e. number of weights = dim_in *
        # dim_out)
        const_term = -0.5 * self.dim_in * self.dim_out

        # Pretty much copied from TensorFlow implementation
        log_std_diff = 0.5 * torch.sum(np.log(self.weight_var_prior) -
                                       self.weight_log_var)
        mu_diff_term = 0.5 * torch.sum((torch.exp(self.weight_log_var) +
                                       (self.weight_mu_prior - self.weight_mu)
                                        / self.weight_var_prior))

        # KL divergence is the sum of these three terms
        kl += const_term + log_std_diff + mu_diff_term

        return kl
