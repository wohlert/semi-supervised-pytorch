import torch
import sys
sys.path.append("../semi-supervised")

cuda = torch.cuda.is_available()
n_labels = 10

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-6) + (1 - x) * torch.log(1 - r + 1e-6), dim=-1)

def get_mnist(location="./", batch_size=64):
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    flatten_bernoulli = lambda img: transforms.ToTensor()(img).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, transform=flatten_bernoulli)
    mnist_valid = MNIST(location, train=False, transform=flatten_bernoulli)

    train = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda)
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda)

    return train, validation

if __name__ == "__main__":
    from torch.autograd import Variable
    from models import LadderVariationalAutoencoder
    from inference import SVI, DeterministicWarmup

    train, validation = get_mnist(batch_size=64)

    model = LadderVariationalAutoencoder([784, [8, 16, 32], [128, 128, 128]])

    objective = SVI(model, beta=DeterministicWarmup(), likelihood=binary_cross_entropy, is_supervised=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.99)

    epochs = 100
    for epoch in range(epochs):

        loss = 0
        for (u, _) in train:
            u = Variable(u)

            if cuda:
                u = u.cuda()

            U = objective(u)
            loss += U.data[0]

            U.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 1 == 0:
            m = len(train)
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t U: {0:.2f}".format(loss / m))

            loss = 0
            for (u, _) in validation:
                u = Variable(u)

                if cuda:
                    u = u.cuda()

                U = objective(u)
                loss += U.data[0]

            m = len(validation)
            print("[Validation]\t U: {0:.2f}".format(loss / m))

            torch.save(model, 'mnist-lvae-{}.pt'.format(epoch))