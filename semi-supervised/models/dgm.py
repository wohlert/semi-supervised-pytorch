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
from torch.nn import init

from .vae import VariationalAutoencoder, Encoder, Decoder


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
        x = F.softplus(self.dense(x))
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
        :param dims: dimensions of x, y, z and hidden layers.
        """
        self.alpha = 0.1 * ratio

        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        logits = self.classifier(x)

        if y is None:
            return logits

        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        # Reconstruct data point from latent data and label
        reconstruction = self.decoder(torch.cat([z, y], dim=1))

        return reconstruction, logits, [[z, z_mu, z_log_var]]

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.type(torch.FloatTensor)
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    """
    M1+M2 model as described in Kingma 2014.
    """
    def __init__(self, dims, ratio, features):
        """
        Initialise a new stacked generative model
        :param ratio: ratio between labelled and unlabelled data
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class VariationalAutoencoder
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim], ratio)

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y=None):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)
