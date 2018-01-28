# Semi-supervised PyTorch

**Work in progress**
**Update January 29th 2018: Several methods have been added now, which will be validated in the coming week.**

A PyTorch-based package containing useful models for modern semi-supervised learning. The models are generally of the deep variety allowing the models to learn in very complex domains.

## What is semi-supervised learning?

Semi-supervised learning tries to bridge the gap between supervised and unsupervised learning by learning from both labelled and unlabelled data.

Semi-supervised learning is typically applied to areas where data is easy to get a hold of, but labelling is expensive. Normally, one would either use an unsupervised method, or just the few labelled examples - both of which would be likely to yield bad results.

The current state-of-the-art method in semi-supervised learning achieves an accuracy of over 99% on the MNIST database using just **10 labelled examples** from each class.

## Implemented models and methods:

* [Variational Autoencoder (Kingma 2013)](https://arxiv.org/abs/1312.6114)
* [Importance Weighted Autoencoders (Burda 2015)](https://arxiv.org/abs/1509.00519)
* [Variational Inference with Normalizing Flows (Rezende & Mohamed 2015)](https://arxiv.org/abs/1505.05770)
* [Semi-supervised Learning with Deep Generative Models (Kingma 2014)](https://arxiv.org/abs/1406.5298)
* [Auxiliary Deep Generative Models (Maaløe 2016)](https://arxiv.org/abs/1602.05473)
* [Ladder Variational Autoencoders (Sønderby 2016)](https://arxiv.org/abs/1602.02282)
