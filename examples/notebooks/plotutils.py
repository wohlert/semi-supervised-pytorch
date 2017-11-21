from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np


def plot_latent(projection, images, labels, fraction):
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