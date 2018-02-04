# Semi-supervised PyTorch

**Update January 29th 2018: Several methods have been added now, which will be validated in the coming week.**

A PyTorch-based package containing useful models for modern deep semi-supervised learning and deep generative models.

## What is semi-supervised learning?

Semi-supervised learning tries to bridge the gap between supervised and unsupervised learning by learning from both
labelled and unlabelled data.

Semi-supervised learning can typically be applied to areas where data is easy to get a hold of, but labelling is expensive.
Normally, one would either use an unsupervised method, or just the few labelled examples - both of which would be
likely to yield bad results.

The current state-of-the-art method in semi-supervised learning achieves an accuracy of over 99% on the MNIST dataset
using just **10 labelled examples** from each class. The table below shows a comparison in accuracy between the models
implemented in this library (achieved) versus their original counterparts (claimed) on MNIST.

| Model              | Claimed accuracy | Achieved accuracy |
| ------------------ | ---------------- | ----------------- |
| Kingma 2014 (M2)   |           88.03% |            93.24% |
| Maaløe 2016 (ADGM) |           99.04% |            95.46% |

## Conditional generation

Most semi-supervised models simultaneously train an inference network and a generator network. This means that it is
not only possible to query this models for classification, but also to generate new data from trained model.
By seperating label information, one can generate a new sample with the given digit as shown in the image below from
Kingma 2014.

![Conditional generation of samples](examples/images/conditional.png)

## Implemented models and methods:

* [Variational Autoencoder (Kingma 2013)](https://arxiv.org/abs/1312.6114)
* [Importance Weighted Autoencoders (Burda 2015)](https://arxiv.org/abs/1509.00519)
* [Variational Inference with Normalizing Flows (Rezende & Mohamed 2015)](https://arxiv.org/abs/1505.05770)
* [Semi-supervised Learning with Deep Generative Models (Kingma 2014)](https://arxiv.org/abs/1406.5298)
* [Auxiliary Deep Generative Models (Maaløe 2016)](https://arxiv.org/abs/1602.05473)
* [Ladder Variational Autoencoders (Sønderby 2016)](https://arxiv.org/abs/1602.02282)
