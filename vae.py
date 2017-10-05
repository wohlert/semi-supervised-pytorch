"""
M1 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def reparametrize(mu, log_var):
    epsilon = Variable(torch.randn(*mu.size()))
    std = torch.exp(0.5 * log_var)

    z = mu + std * epsilon

    return z

class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        self.dense1 = nn.Linear(x_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, 2*z_dim)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        self.dense1 = nn.Linear(z_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder model consisting
    of an encoder/decoder pair for which a
    variational distribution is fitted to the
    encoder.
    """
    def __init__(self, dims):
        super(VariationalAutoencoder, self).__init__()
        [x_dim, z_dim, h_dim] = dims
        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, h_dim, x_dim])

    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=1)
        z = reparametrize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, (mu, log_var)

    def generate(self, z):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param y: label
        :return: x
        """
        return self.decoder(z)


def train_vae(model, dataloader, optimizer, epochs=100):
    """
    Trains a Variational Autoencoder model.
    :param model: object of class `VariationalAutoencoder`
    :param dataloader: loader to sample data from
    :param optimizer: PyTorch compatible optimizer
    :param epochs: number of epochs to run optimization over
    """
    for epoch in range(epochs):
        for input, _ in dataloader:
            input = input.view(dataloader.batch_size, -1)
            input = Variable(input)

            optimizer.zero_grad()

            reconstruction, (mu, log_var) = model(input)

            loss = F.binary_cross_entropy(reconstruction, input, size_average=False)
            kl_divergence = 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1.)

            total_loss = loss + kl_divergence

            total_loss.backward()
            optimizer.step()

        print("Epoch {0:}, loss: {1:.2f}, KL: {2:.2f}".format(epoch, loss, kl_divergence))