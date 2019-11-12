import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

from vcl_modules import BayesianLinear
from scipy.stats import truncnorm
from copy import deepcopy

class MFVINN(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_size, training_size,
                 n_train_samples=10):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        # Number of samples to take when estimating log-likelihood
        self.n_train_samples = n_train_samples
        # Number of training data
        self.training_size = training_size

        hidden_size = deepcopy(hidden_size)
        # Include first and last layers
        hidden_size.append(dim_out)
        hidden_size.insert(0, dim_in)

        self.size = hidden_size
        self.n_layers = len(self.size) - 1

        # Create empty ModuleList for storing BayesianLinear layers
        self.layers = nn.ModuleList()

        for i in range(len(self.size) - 2):
            # Input and output dimensions for each layer
            din = self.size[i]
            dout = self.size[i+1]

            layer = BayesianLinear(din, dout)

            self.layers.append(layer)

        # Create empty ModuleList for storing last BayesianLinear layers.
        # There will be one last layer for each task, accessible via task_idx.
        self.last_layers = nn.ModuleList()

        # Initialise the first head.
        self.create_head()

    def get_loss(self, inputs, targets, task_idx):
        # Divide KL term by number of training data
        return torch.div(self._KL_term(), self.training_size) - \
               self._logpred(inputs, targets, task_idx)

    def forward(self, inputs, task_idx):
        x = inputs.view(-1, self.dim_in)
        # Propagate through shared layers first
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        # Propagate through task specific layer
        x = self.last_layers[task_idx](x)

        return x

    def _logpred(self, inputs, targets, task_idx):
        loss = torch.nn.CrossEntropyLoss()
        # Get predictions by sampling weights from each layer.
        log_lik = 0
        for i in range(self.n_train_samples):
            preds_sample = self.forward(inputs, task_idx)
            log_lik_sample = loss(preds_sample, targets.type(torch.long))
            log_lik += log_lik_sample

        # Calculate Monte-Carlo estimate of log-likelihood
        log_lik = - log_lik / self.n_train_samples

        return log_lik



    def _KL_term(self):
        kl = 0

        # Loop through shared layers
        for layer in self.layers:
            kl += layer.KL_loss()

        # Loop through last layers
        for last_layer in self.last_layers:
            kl += last_layer.KL_loss()

        return kl

    def create_head(self):
        # Create a new last layer and append to the end of last_layers
        # ModuleList.
        din = self.size[-2]
        dout = self.size[-1]
        layer = BayesianLinear(din, dout)

        self.last_layers.append(layer)

        return

    def update_prior(self):
        # Loop through shared layers.
        for layer in self.layers:
            # Update prior mean and variances to previous posterior values
            layer.weight_mu_prior.data.copy_(
                layer.weight_mu.clone().detach().data)

            layer.weight_var_prior.data.copy_(
                torch.exp(layer.weight_log_var.clone().detach().data))

            layer.bias_mu_prior.data.copy_(layer.bias_mu.clone().detach().data)

            layer.bias_var_prior.data.copy_(
                torch.exp(layer.bias_log_var.clone().detach().data))

        # Loop through task specific last layers
        for last_layer in self.last_layers:
            # Update prior mean and variances to previous posterior values
            last_layer.weight_mu_prior.data.copy_(
                last_layer.weight_mu.clone().detach().data)

            last_layer.weight_var_prior.data.copy_(
                torch.exp(last_layer.weight_log_var.clone().detach().data))

            last_layer.bias_mu_prior.data.copy_(
                last_layer.bias_mu.clone().detach().data)

            last_layer.bias_var_prior.data.copy_(
                torch.exp(last_layer.bias_log_var.clone().detach().data))

        return

class MFVINNWrapper():
    def __init__(self, dim_in, dim_out, hidden_size, learn_rate, training_size,
                 device):

        # Set device for MFVINNWrapper
        self.device = device

        # Initialise MFVINN object, and move to device
        self.mfvi_net = MFVINN(dim_in, dim_out, hidden_size, training_size
                               ).to(self.device)

        # Initialise ADAM optimiser.
        self.optimizer = optim.Adam(self.mfvi_net.parameters(), lr=learn_rate)

    def train(self, x_train, y_train, task_idx, n_epochs=1000, batch_size=100,
              display_epoch=5):

        # Get total number of training data points.
        N = x_train.shape[0]
        self.mfvi_net.training_size = N

        if batch_size > N:
            batch_size = N

        losses = []
        # Training cycle
        for e in range(n_epochs):
            # Randomly shuffle training data for each epoch
            perm_inds = np.arange(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_loss = 0
            n_batches = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(n_batches):
                # Start index
                start_ind = i * batch_size
                # End index
                end_ind = np.min([(i + 1) * batch_size, N])

                # Extract batch from training data
                x_batch = torch.Tensor(cur_x_train[start_ind:end_ind]
                                       ).to(self.device)
                y_batch = torch.Tensor(cur_y_train[start_ind:end_ind]
                                       ).to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()
                # Calculate the loss for given batch
                loss = self.mfvi_net.get_loss(x_batch, y_batch, task_idx)
                # Calculate gradients and update parameter values
                loss.backward()
                self.optimizer.step()

                # Compute average loss
                avg_loss += loss / n_batches

            # Display logs per epoch step
            if e % display_epoch == 0:
                print('Epoch: {:04d} Loss: {:.9f}'.format(e + 1, avg_loss))

            losses.append(avg_loss)

        print('Optimisation Finished!')

        return losses

    def create_head(self):
        self.mfvi_net.create_head()
        return

    def update_prior(self):
        self.mfvi_net.update_prior()
        return
