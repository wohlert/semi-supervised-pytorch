# Semi-supervised PyTorch

**Work in progress**.

A PyTorch-based package containing useful models for modern semi-supervised learning. The models are generally of the deep variety allowing the models to learn in very complex domains.

## What is semi-supervised learning?

Semi-supervised learning tries to bridge the gap between supervised and unsupervised learning by learning from both labelled and unlabelled data.

Semi-supervised learning is typically applied to areas where data is easy to get a hold of, but labelling is expensive. Normally, one would either use an unsupervised method, or just the few labelled examples - both of which would be likely to yield bad results.

The current state-of-the-art method in semi-supervised learning achieves an accuracy of over 99% on the MNIST database using just **10 labelled examples** from each class.

The image below shows the training procedure in [visdom](https://github.com/facebookresearch/visdom) giving an accuracy of 89% on the first five digits only using 50 labels in total.

![Training DGM in visdom](examples/images/visdom-dgm.png)

## Implemented models:

* [Variational Autoencoder (Kingma 2013)](https://arxiv.org/abs/1312.6114)
* [Semi-supervised Learning with Deep Generative Models (Kingma 2014)](https://arxiv.org/abs/1406.5298)
* [Auxiliary Deep Generative Models (Maaløe 2016)](https://arxiv.org/abs/1602.05473)

## Planned models and methods

* [Ladder Variational Autoencoders (Sønderby 2016)](https://arxiv.org/abs/1602.02282)
* [Importance Weighted Autoencoders (Burda 2015)](https://arxiv.org/abs/1509.00519)
