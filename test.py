from pathlib import Path

import matplotlib.pyplot as plt

from src.data.dataset import Pix2PixDataset

if __name__ == '__main__':
    data_dir = Path('./cats_dataset/train')
    dataset = Pix2PixDataset(data_dir)

    for x, y in dataset:
        plt.imshow(x.numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.show()
        plt.imshow(y.numpy().transpose((1, 2, 0)) * 0.5 + 0.5)
        plt.show()
        break
