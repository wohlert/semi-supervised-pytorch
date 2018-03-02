from urllib import request

import torch
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append("../semi-supervised")

torch.manual_seed(1337)
np.random.seed(1337)

cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


class SpriteDataset(Dataset):
    def __init__(self, transform=None, download=False):
        self.transform = transform
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

        if download:
            request.urlretrieve(url, "./dsprites.npz")

        try:
            self.dset = np.load("./dsprites.npz", encoding="bytes")["imgs"]
        except FileNotFoundError:
            print("Dataset not found, have you set download=True?")

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        sample = self.dset[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    from itertools import repeat
    from torch.autograd import Variable

    dset = SpriteDataset(transform=lambda x: x.reshape(-1), download=True)
    unlabelled = DataLoader(dset, batch_size=16, shuffle=True, sampler=SubsetRandomSampler(np.arange(len(dset)//3)))

    models = []

    from models import VariationalAutoencoder
    model = VariationalAutoencoder([64**2, 10, [1200, 1200]])
    model.decoder = nn.Sequential(
        nn.Linear(10, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(10, 64**2),
        nn.Sigmoid(),
    )

    if cuda: model = model.cuda()

    beta = repeat(4.0)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)

    epochs = 251
    best = np.inf

    file = open(model.__class__.__name__ + ".log", 'w+')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u in unlabelled:
            u = Variable(u.float())

            if cuda:
                u = u.cuda(device=0)

            reconstruction = model(u)

            likelihood = -binary_cross_entropy(reconstruction, u)
            elbo = likelihood - next(beta) * model.kl_divergence

            L = -torch.mean(elbo)

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += L.data[0]

        m = len(unlabelled)
        print(total_loss / m, sep="\t")

        if total_loss < best:
            best = total_loss
            torch.save(model, '{}.pt'.format(model.__class__.__name__))
