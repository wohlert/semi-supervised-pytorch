import torch
import numpy as np
import sys
sys.path.append("../semi-supervised")

cuda = torch.cuda.is_available()
print("CUDA:", cuda)
n_labels = 10

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

def get_mnist(location="./", batch_size=64, labels_per_class=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    flatten_bernoulli = lambda img: transforms.ToTensor()(img).view(-1).bernoulli()

    def one_hot(n):
        def encode(label):
            y = torch.zeros(n)
            if label < n:
                y[label] = 1
            return y
        return encode

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=one_hot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=one_hot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation

if __name__ == "__main__":
    from itertools import repeat, cycle
    from torch.autograd import Variable
    from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler

    labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)
    alpha = 0.1 * len(unlabelled) / len(labelled)

    models = []

    # Kingma 2014, M2 model. Reported: 88%, achieved: ??%
    from models import DeepGenerativeModel
    models += [DeepGenerativeModel([784, n_labels, 50, [600, 600]])]

    # Maaløe 2016, ADGM model. Reported: 99.4%, achieved: ??%
    from models import AuxiliaryDeepGenerativeModel
    models += [AuxiliaryDeepGenerativeModel([784, n_labels, 100, 100, [500, 500]])]

    from models import LadderDeepGenerativeModel
    models += [LadderDeepGenerativeModel([784, n_labels, [64, 32, 16, 8, 4], [512, 256, 128, 64, 32]])]

    from models import LadderDeepGenerativeModel2
    models += [LadderDeepGenerativeModel2([784, n_labels, [64, 32, 16, 8, 4], [512, 256, 128, 64, 32]])]

    from models import LadderDeepGenerativeModel3
    models += [LadderDeepGenerativeModel3([784, n_labels, [64, 32, 16, 8, 4], [512, 256, 128, 64, 32]])]

    for model in models:
        if cuda: model = model.cuda()

        if type(model) == DeepGenerativeModel:
            beta = repeat(1)
            sampler = ImportanceWeightedSampler(1, 1)
        else:
            beta = DeterministicWarmup()
            sampler = ImportanceWeightedSampler(1, 5)

        elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

        epochs = 251
        best = 0.0

        file = open(model.__class__.__name__ + ".log", 'w+')

        for epoch in range(epochs):
            total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
            for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
                # Wrap in variables
                x, y, u = Variable(x), Variable(y), Variable(u)

                if cuda:
                    x, y, u = x.cuda(), y.cuda(), u.cuda()

                L = -elbo(x, y)
                U = -elbo(u)

                # Add auxiliary classification loss q(y|x)
                logits = model.classify(x)
                classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                J_alpha = L - alpha * classication_loss + U

                J_alpha.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += J_alpha.data[0]
                labelled_loss += L.data[0]
                unlabelled_loss += U.data[0]

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

            m = len(unlabelled)
            print(*(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m), sep="\t", file=file)

            if epoch % 1 == 0:
                print("Epoch: {}".format(epoch))
                print("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                              labelled_loss / m,
                                                                                              unlabelled_loss / m,
                                                                                              accuracy / m))

                total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                for x, y in validation:
                    x, y = Variable(x), Variable(y)

                    if cuda:
                        x, y = x.cuda(), y.cuda()

                    L = -elbo(x, y)
                    U = -elbo(x)

                    logits = model.classify(x)
                    classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                    J_alpha = L - alpha * classication_loss + U

                    total_loss += J_alpha.data[0]
                    labelled_loss += L.data[0]
                    unlabelled_loss += U.data[0]

                    _, pred_idx = torch.max(logits, 1)
                    _, lab_idx = torch.max(y, 1)
                    accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                m = len(validation)
                print(*(total_loss / m, labelled_loss / m, unlabelled_loss / m, accuracy / m), sep="\t", file=file)
                print("[Validation]\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                              labelled_loss / m,
                                                                                              unlabelled_loss / m,
                                                                                              accuracy / m))

            if accuracy > best:
                best = accuracy
                torch.save(model, '{}.pt'.format(model.__class__.__name__))