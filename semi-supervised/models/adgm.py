"""
Auxiliary Deep Generative Models (Maaløe 2016)
code replication.
"""

import torch
import torch.nn as nn

from .vae import Encoder, Decoder, VariationalAutoencoder
from .dgm import Classifier

class AuxiliaryDeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, ratio, dims):
        self.alpha = 0.1
        self.beta = self.alpha * ratio

        [self.x_dim, self.y_dim, self.z_dim, self.h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([self.x_dim, self.z_dim, self.h_dim])

        self.aux_encoder = Encoder([self.x_dim, self.h_dim, self.z_dim])
        self.encoder = Encoder([self.z_dim, self.h_dim, self.z_dim])

        self.aux_decoder = Decoder([self.z_dim, list(reversed(self.h_dim)), self.z_dim])
        self.decoder = Decoder([self.z_dim, list(reversed(self.h_dim)), self.x_dim])

        self.classifier = Classifier([self.x_dim, self.h_dim[0], self.y_dim])

        # Transform layers
        self.transform_x_to_z = nn.Linear(self.x_dim, self.z_dim)
        self.transform_y_to_z = nn.Linear(self.y_dim, self.z_dim)
        self.transform_z_to_x = nn.Linear(self.z_dim, self.x_dim)

    def forward(self, x, y=None):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction, logits, [z], [a]
        """
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(self.transform_z_to_x(a) + x)

        if y is None:
            return logits

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(a + self.transform_y_to_z(y) + self.transform_x_to_z(x))

        # Generative p(a|z,y)
        a = self.aux_decoder(z + self.transform_y_to_z(y))

        # Generative p(x|a,z,y)
        reconstruction = self.decoder(a + z + self.transform_y_to_z(y))

        return reconstruction, logits, [z, z_mu, z_log_var], [a, a_mu, a_log_var]

    def sample(self, z, a, y):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param a: Auxiliary normal variable
        :param y: label
        :return: x
        """
        a = a.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        y = self.transform_y_to_z(y)
        return self.decoder(z + a + y)
