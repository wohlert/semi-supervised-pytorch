import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder, LadderEncoder
from layers import GaussianMerge


class Classifier(nn.Module):
    """
    Single hidden layer classifier
    with softmax output.
    """
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    """
    M2 code replication from the paper
    'Semi-Supervised Learning with Deep Generative Models'
    (Kingma 2014) in PyTorch.

    The "Generative semi-supervised model" is a probabilistic
    model that incorporates label information in both
    inference and generation.
    """
    def __init__(self, dims):
        """
        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
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

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    """
    M1+M2 model as described in Kingma 2014.
    """
    def __init__(self, dims, features):
        """
        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim])

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    """
    Auxiliary Deep Generative Models [Maaløe 2016]
    code replication. The ADGM introduces an additional
    latent variable 'a', which enables the model to fit
    more complex variational distributions.
    """
    def __init__(self, dims):
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([a, x], dim=1))
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([q_a, x, y], dim=1))

        # Generative p(x|z,y)
        x_mu = self.decoder(torch.cat([y, z], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu

    def sample(self, z, a, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param a: auxiliary normal variable
        :param y: label
        :return: x
        """
        return super(AuxiliaryDeepGenerativeModel, self).sample(torch.cat([a, z], dim=1), y)


class LadderDeepGenerativeModel(DeepGenerativeModel):
    """
    Semi-supervised ladder variational autoencoder.
    """
    def __init__(self, dims, config=2):
        [x_dim, y_dim, z_dim, h_dim] = dims
        self.config = config
        super(LadderDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        ladder_layers = [LadderEncoder([neurons[i - 1], [neurons[i]], z_dim[i - 1], 0]) for i in range(1, len(neurons))]

        if self.config == 0:
            l = ladder_layers[0]
            ladder_layers[0] = LadderEncoder([l.in_features, l.out_features, l.z_dim, y_dim])

        self.encoder = nn.ModuleList(ladder_layers)
        self.merge = nn.ModuleList(
            [GaussianMerge(z_dim[len(z_dim) - i], z_dim[len(z_dim) - i - 1]) for i in range(1, len(z_dim))])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == 0:
                x, (z, mu, log_var) = encoder(x, y)
            else:
                x, (z, mu, log_var) = encoder(x)
            latents.append((z, mu, log_var))

        self.kl_divergence = 0
        for i, latent in enumerate(reversed(latents)):
            # If at top, encoder == decoder,
            # use prior for KL.
            q_z, q_mu, q_log_var = latent
            if i == 0:
                self.kl_divergence += self._kld(q_z, (q_mu, q_log_var))

            # Perform downword merge of information.
            else:
                z, p_mu, p_log_var = self.merge[i - 1](z, q_mu, q_log_var )
                self.kl_divergence += self._kld(q_z, (q_mu, q_log_var), (p_mu, p_log_var))

        x_mu = self.decoder(torch.cat([z, y], dim=1))
        return x_mu