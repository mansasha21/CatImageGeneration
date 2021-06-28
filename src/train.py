import argparse
import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import Pix2PixDataset
from src.model.discriminator import Discriminator
from src.model.generator import Generator
from src.trainer.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='../data_examples/train/')
    parser.add_argument('--val_dir', default=None, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--checkpoint_dir', default='../models/')
    parser.add_argument('--checkpoint_freq', default=10, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--l1_coef', default=100, type=int)

    args = parser.parse_args()

    model = {'gen': Generator().to(args.device),
             'disc': Discriminator().to(args.device)}

    train_dataset = Pix2PixDataset(Path(args.train_dir))
    train_loader = DataLoader(train_dataset, shuffle=True)

    val_loader = None
    if args.val_dir is not None:
        val_dataset = Pix2PixDataset(Path(args.val_dir))
        val_loader = DataLoader(val_dataset, shuffle=False)

    optimizers = {
        'gen': torch.optim.Adam(model['gen'].parameters(), lr=args.lr, betas=(0.5, 0.999)),
        'disc': torch.optim.Adam(model['disc'].parameters(), lr=args.lr, betas=(0.5, 0.999))
    }

    trainer = Trainer(model, optimizers, train_loader, args.device, args.checkpoint_dir, val_loader)
    history = trainer.fit(args.max_epochs, args.l1_coef, args.checkpoint_freq)

    file_name = '../logs/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(file_name, 'w+') as file:
        json.dump(history, file)









