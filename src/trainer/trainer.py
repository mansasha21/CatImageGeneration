from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from IPython.display import clear_output
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizers, dataloader, device, save_dir, val_loader=None):
        self.model = model
        self.optimizers = optimizers
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.save_dir = Path(save_dir)

    def fit(self, epochs=1, l1_coef=100, save_freq=4):
        history = {'gen_loss': [], 'disc_loss': [], 'val_gen_loss': [], 'val_disc_loss': []}

        self.model['disc'].train()
        self.model['gen'].train()
        for epoch in range(epochs):
            history_epoch = self._train_epoch(l1_coef)
            for key, arr in history_epoch.items():
                history[key].append(np.mean(arr))
            if self.val_loader is not None:
                val_history_epoch = self._validate(l1_coef)
                for key, arr in val_history_epoch.items():
                    history[key].append(np.mean(arr))

            clear_output(wait=True)
            print('[%d] gen_loss: %.3f, disc_loss: %.3f, '
                  % (epoch + 1, history['gen_loss'][-1], history['disc_loss'][-1]))
            if self.val_loader is not None:
                print('[%d] val_gen_loss: %.3f, val_disc_loss: %.3f, '
                      % (epoch + 1, history['val_gen_loss'][-1], history['val_disc_loss'][-1]))

            if (epoch + 1) % save_freq == 0:
                file_gen = self.save_dir / 'cat_gen_{}.state'.format(epoch)
                file_disc = self.save_dir / 'cat_disc_{}.state'.format(epoch)
                torch.save(self.model['gen'].state_dict(), file_gen)
                torch.save(self.model['disc'].state_dict(), file_disc)

        return history

    def _train_epoch(self, l1_coef):
        history_epoch = {'gen_loss': [], 'disc_loss': []}
        for x_batch, y_batch in tqdm(self.dataloader, total=len(self.dataloader)):
            x_batch = x_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()

            self.optimizers['disc'].zero_grad()
            disc_loss = self._calc_disc_loss(x_batch, y_batch)
            disc_loss.backward()
            self.optimizers['disc'].step()
            history_epoch['disc_loss'].append(disc_loss.item() * x_batch.size(0))

            self.optimizers['gen'].zero_grad()
            gen_loss = self._calc_gen_loss(x_batch, y_batch, l1_coef)
            gen_loss.backward()
            self.optimizers['gen'].step()

            history_epoch['gen_loss'].append(gen_loss.item() * x_batch.size(0))

        return history_epoch

    def _validate(self, l1_coef):
        history = {'val_gen_loss': [], 'val_disc_loss': []}

        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.val_loader, total=len(self.val_loader)):
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()
                disc_loss = self._calc_disc_loss(x_batch, y_batch)
                gen_loss = self._calc_gen_loss(x_batch, y_batch, l1_coef)
                history['val_gen_loss'].append(gen_loss.item() * x_batch.size(0))
                history['val_disc_loss'].append(disc_loss.item() * x_batch.size(0))
        return history

    def _calc_disc_loss(self, x_batch, y_batch):
        gen_images = self.model['gen'](x_batch)
        real_preds = self.model['disc'](x_batch, y_batch)
        disc_real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
        gen_preds = self.model['disc'](x_batch, gen_images.detach())
        disc_gen_loss = self.bce_loss(gen_preds, torch.zeros_like(gen_preds))
        disc_loss = (disc_real_loss + disc_gen_loss) / 2
        return disc_loss

    def _calc_gen_loss(self, x_batch, y_batch, l1_coef):
        gen_images = self.model['gen'](x_batch)
        gen_preds = self.model['disc'](x_batch, gen_images.detach())
        gen_gen_loss = self.bce_loss(gen_preds, torch.zeros_like(gen_preds))
        additional_loss = self.l1_loss(gen_images, y_batch) * l1_coef
        return gen_gen_loss + additional_loss

