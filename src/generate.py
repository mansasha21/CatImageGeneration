import argparse
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from src.data.dataset import Pix2PixGenDataset
from src.model.generator import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_dir', default='../data_examples/generate/')
    parser.add_argument('--gen_path', default='../models/face_catgen2_44.state')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_dir', default='../data_examples/generated_imgs/')
    args = parser.parse_args()

    generator = Generator().to(args.device)
    generator.load_state_dict(torch.load(args.gen_path))
    gen_dataset = Pix2PixGenDataset(Path(args.edge_dir))
    gen_loader = DataLoader(gen_dataset)
    with torch.no_grad():
        for i, x in enumerate(gen_loader):
            x = x.to(args.device)
            gen_img = ((generator(x.float())[0].cpu().numpy()
                       .transpose((1, 2, 0)) * 0.5 + 0.5)[..., [2, 1, 0]] * 255).astype(int)
            save_path = str(Path(args.save_dir) / (str(i) + '.jpg'))
            cv2.imwrite(save_path, gen_img)
            del gen_img

    del generator
    torch.cuda.empty_cache()









