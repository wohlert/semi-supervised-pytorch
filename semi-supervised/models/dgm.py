"""
M2 code replication from the paper
'Semi-Supervised Learning with Deep Generative Models'
(Kingma 2014) in PyTorch.

The "Generative semi-supervised model" is a probabilistic
model that incorporates label information in both
inference and generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from utils import generate_label
from .vae import VariationalAutoencoder, Encoder, Decoder


# Shorthand cross_entropy till fix in next version of PyTorch
def cross_entropy(logits, y):
    return -torch.sum(y * torch.log(logits + 1e-8), dim=1)


class Classifier(nn.Module):
    """
    Two layer classifier
    with softmax output.
    """
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)
        self.output_activation = F.softmax

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = self.output_activation(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    """
    Deep generative model consisting of an
    encoder/decoder pair along with a classifier
    in order to perform semi-supervised learning.
    """
    def __init__(self, dims, ratio):
        """
        Initialise a new generative model
        :param ratio: ratio between labelled and unlabelled data
        :param dims: dimensions of x, y, z and hidden layer.
        """
        self.alpha = 0.1
        self.beta = self.alpha * ratio

        [self.x_dim, self.y_dim, self.z_dim, self.h_dim] = dims
        super(DeepGenerativeModel, self).__init__([self.x_dim, self.z_dim, self.h_dim])

        self.encoder = Encoder([self.h_dim[0], self.h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim, list(reversed(self.h_dim)), self.x_dim])
        self.classifier = Classifier([self.x_dim, self.h_dim[0], self.y_dim])

        # Transform layers
        self.transform_x_to_h = nn.Linear(self.x_dim, self.h_dim[0])
        self.transform_y_to_h = nn.Linear(self.y_dim, self.h_dim[0])
        self.transform_y_to_z = nn.Linear(self.y_dim, self.z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        # Classify the data point
        logits = self.classifier(x)

        if y is None:
            return logits

        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(self.transform_x_to_h(x) + self.transform_y_to_h(y))

        # Reconstruct data point from latent data and label
        reconstruction = self.decoder(z + self.transform_y_to_z(y))

        return reconstruction, logits, [[z, z_mu, z_log_var]]

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.type(torch.FloatTensor)
        y = self.transform_y_to_z(y)
        x_mu = self.decoder(z + y)
        return x_mu
