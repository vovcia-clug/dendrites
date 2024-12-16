import numpy
import numpy as np
from matplotlib import pyplot, pyplot as plt


def log_matshow(tensors, writer, step, titles=None):
    num_tensors = tensors.shape[0]
    fig, axs = plt.subplots(num_tensors, 1, figsize=(20, 2 * num_tensors))
    if titles is None:
        titles = [f"Plot {i + 1}" for i in range(num_tensors)]

    for i in range(num_tensors):
        if num_tensors > 1:
            ax = axs[i]
        else:
            ax = axs
        aspect_ratio = 10
        cax = ax.imshow(tensors[i].T, cmap='jet', aspect=aspect_ratio)
        cax.set_clim(-1.0, 1.0)

        if i == 0:
            ax.set_title(titles[i])

    # Convert plot to tensor and log to TensorBoard
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = np.moveaxis(image, 2, 0)  # PyTorch expects CxHxW
    writer.add_image(f'Voltage {titles[0]}', image, global_step=step)
    plt.show()
    plt.close(fig)
