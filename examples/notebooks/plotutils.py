from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np


def latent_scatter(projection, images, labels, fraction):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.scatter(*projection.T, c=labels, cmap="viridis")

    min_dist_2 = (0.1 * max(projection.max(0) - projection.min(0))) ** 2
    shown_images = np.array([2 * projection.max(0)])
    for i in range(len(images)):
        dist = np.sum((projection[i] - shown_images)**2, 1)
        if np.min(dist) < min_dist_2:
            # don't show points that are too close
            continue
        shown_images = np.vstack([shown_images, projection[i]])
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i]), projection[i])
        ax.add_artist(imagebox)
        
    return ax


def latent_imshow(z_min, z_max, n, model):
    size = 28
    x = y = np.linspace(z_min, z_max, n)
    grid = np.zeros(shape=(size*n, size*n))

    for i, z_x in enumerate(x):
        for j, z_y in enumerate(y):
            z = Variable(torch.FloatTensor([z_x, z_y]))
            sample = model.sample(z).cpu()
            sample = sample.data.cpu().numpy().reshape(size, size)
            grid[size*i:size*(i+1), size*j:size*(j+1)] = sample

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    plt.imshow(grid, extent=[z_min, z_max, z_min, z_max])

    plt.title('VAE latent space representation', fontsize = 14)
    plt.xlabel('z dimension 1', fontsize = 14)
    plt.ylabel('z dimension 2', fontsize = 14)