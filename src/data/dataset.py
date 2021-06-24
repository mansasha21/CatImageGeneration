from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset

IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
MEAN = 0.5
STD = 0.5
MAX_PIXEL_VALUE = 255.


class Pix2PixDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.files = list(data_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        image = np.array(Image.open(self.files[item]))

        image = (image / MAX_PIXEL_VALUE - MEAN) / STD
        x, y = image[:, :image.shape[1] // 2], image[:, image.shape[1] // 2:]
        x = transform.resize(x, (IMAGE_SIZE, IMAGE_SIZE)).transpose((2, 0, 1))
        y = transform.resize(y, (IMAGE_SIZE, IMAGE_SIZE)).transpose((2, 0, 1))

        return torch.tensor(x), torch.tensor(y)


