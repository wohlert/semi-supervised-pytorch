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
from torch.nn import init

from layers import GaussianSample, GaussianMerge
from inference import log_gaussian, log_standard_gaussian


class Encoder(nn.Module):
    """
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
        self.sample = GaussianSample(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    """
    Generative network

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
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder model consisting
    of an encoder/decoder pair for which a
    variational distribution is fitted to the
    encoder.

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, dims):
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kl_divergence(self, z, param1, param2=None):
        """
        Computes the KL-divergence of for
        some element z.
        :param z:
        :param param1:
        :param param2:
        :return:
        """
        if param2 is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = param2
            pz = log_gaussian(z, mu, log_var)

        (mu, log_var) = param1
        qz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters

        :param x (torch.autograd.Variable): input data
        :return: (x_hat, latent) where latent is represented
        by parameters of the z-distribution along with a
        sample
        """
        z, z_mu, z_log_var = self.encoder(x)

        kl = self._kl_divergence(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu, kl

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class LadderEncoder(nn.Module):
    """
    The ladder encoder differs from the standard encoder
    by using batch-normalization and ELU activation.
    Additionally, it also returns the transformation x.
    """
    def __init__(self, dims):
        super(LadderEncoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(l.out_features) for l in self.hidden])
        self.sample = GaussianSample(h_dim[-1], z_dim)

    def forward(self, x):
        for linear, batch in zip(self.hidden, self.batchnorm):
            x = linear(x)
            x = F.elu(batch(x))
        return x, self.sample(x)


class LadderVariationalAutoencoder(VariationalAutoencoder):
    """
    Ladder Variational Autoencoder as described by
    Sønderby et al (2016). Adds several stochastic
    layers to improve the log-likelihood estimate.

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], [latent_dims]].
    """
    def __init__(self, dims):
        [x_dim, z_dim, h_dim] = dims
        super(LadderVariationalAutoencoder, self).__init__([x_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        ladder_layers = [LadderEncoder([neurons[i-1], [neurons[i]], z_dim[i-1]]) for i in range(1, len(neurons))]

        self.encoder = nn.ModuleList(ladder_layers)
        self.merge = [GaussianMerge(z_dim[len(z_dim)-i], z_dim[len(z_dim)-i-1]) for i in range(1, len(z_dim))]

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for encoder in self.encoder:
            x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        kl = 0
        for i, latent in enumerate(reversed(latents)):
            # If at top, encoder == decoder,
            # use prior for KL.
            if i == 0:
                kl += self._kl_divergence(z, latent)

            # Perform downword merge of information.
            else:
                z, mu, log_var = self.merge[i-1](z, *latent)
                kl += self._kl_divergence(z, (mu, log_var), latent)

        x_mu = self.decoder(z)
        return x_mu, kl