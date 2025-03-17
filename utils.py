import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_images(data, labels, class_names=None, flatten=False, use_wandb=False):

    uniq_labels = np.unique(labels)

    fig, ax = plt.subplots(2,5, figsize=(15, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = ax.reshape(-1)

    for i, label in enumerate(uniq_labels):
        img = data[np.where(labels == label)[0][0]]
        if class_names:
            ax[i].set_title(class_names[label])
        if flatten:
            img = img.reshape(28, 28)
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    if use_wandb:
        wandb.log({"Class Images": fig})
    plt.close(fig)