"""
M1 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.

This "Latent-feature discriminative model" is eqiuvalent
to a classifier with VAE latent representation as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    r"""
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.mu = nn.Linear(h_dim[-1], z_dim)
        self.log_var = nn.Linear(h_dim[-1], z_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden) - 1:
                x = F.relu(x)
        return self.mu(x), self.log_var(x)


class Decoder(nn.Module):
    r"""
    Generator network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = F.relu(layer(x))
        x = self.output_activation(self.reconstruction(x))
        return x


class VariationalAutoencoder(nn.Module):
    r"""
    Variational Autoencoder model consisting
    of an encoder/decoder pair for which a
    variational distribution is fitted to the
    encoder.

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    :return: (x_hat, latent) where latent is represented
        by parameters of the z-distribution along with a
        sample.
    """
    def __init__(self, dims):
        super(VariationalAutoencoder, self).__init__()
        [x_dim, z_dim, h_dim] = dims
        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, (z, mu, log_var)

    def generate(self, z):
        r"""
        Given z ~ N(0, I) generates a sample from
        the learning distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def reparametrize(self, mu, log_var):
        r"""
        Performs the reparametrisation trick as described
        by (Kingma 2013) in order to backpropagate through
        stochastic units.
        :param mu: (torch.autograd.Variable) mean of normal distribution
        :param log_var: (torch.autograd.Variable) log variance of normal distribution
        :return: (torch.autograd.Variable) a sample from the distribution
        """
        epsilon = Variable(torch.randn(*mu.size()))
        std = torch.exp(0.5 * log_var)

        z = mu + std * epsilon

        return z