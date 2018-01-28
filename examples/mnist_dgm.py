import torch
import numpy as np
import sys
sys.path.append("../semi-supervised")

cuda = torch.cuda.is_available()
n_labels = 10

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-6) + (1 - x) * torch.log(1 - r + 1e-6), dim=-1)

def get_mnist(location="./", batch_size=64, n_labelled=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    flatten_bernoulli = lambda img: transforms.ToTensor()(img).view(-1).bernoulli()

    def one_hot(n):
        def encode(label):
            y = torch.zeros(n)
            y[label] = 1
            return y
        return encode

    mnist_train = MNIST(location, train=True,
                        transform=flatten_bernoulli, target_transform=one_hot(n_labels))
    mnist_valid = MNIST(location, train=False,
                        transform=flatten_bernoulli, target_transform=one_hot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        indices = [np.where(labels[indices] == i)[:n] for i in range(n_labels)]

        indices = torch.from_numpy(np.hstack(indices).ravel())
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), n_labelled))

    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation

if __name__ == "__main__":
    from itertools import repeat, cycle
    from torch.autograd import Variable
    from models import DeepGenerativeModel
    from inference import SVI

    labelled, unlabelled, validation = get_mnist(batch_size=64, n_labelled=2)

    model = DeepGenerativeModel([784, n_labels, 16, [128]], len(unlabelled) / len(labelled))

    objective = SVI(model, beta=repeat(1), likelihood=binary_cross_entropy, is_supervised=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.99)

    epochs = 100
    for epoch in range(epochs):

        labelled_loss, unlabelled_loss, accuracy = (0, 0, 0)
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
            x = Variable(x)
            y = Variable(y).float()
            u = Variable(u)

            if cuda:
                x = x.cuda()
                y = y.cuda()
                u = u.cuda()

            L = objective(x, y)
            U = objective(u)

            J = L + U

            labelled_loss += L.data[0]
            unlabelled_loss += U.data[0]

            _, pred_idx = torch.max(model.classifier(x), 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

            J.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1 == 0:
            m = len(unlabelled)
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t L: {0:.2f}, U: {1:.2f}, accuracy: {2:.2f}".format(labelled_loss / m, unlabelled_loss / m, accuracy / m))

            labelled_loss, unlabelled_loss, accuracy = (0, 0, 0)
            for (x, y) in validation:
                x = Variable(x)
                y = Variable(y).float()

                if cuda:
                    x = x.cuda()
                    y = y.cuda()

                L = objective(x, y)
                U = objective(x)

                labelled_loss += L.data[0]
                unlabelled_loss += U.data[0]

                _, pred_idx = torch.max(model.classifier(x), 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

            m = len(validation)
            print("[Validation]\t L: {0:.2f}, U: {1:.2f}, accuracy: {2:.2f}".format(labelled_loss / m, unlabelled_loss / m, accuracy / m))

            torch.save(model, 'mnist-dgm-{}.pt'.format(epoch))