import data

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class RNNDynamics(pl.LightningModule):

    def __init__(self, data_dim, hidden_size):
        super().__init__()
        self.d = data_dim
        self.n = hidden_size
        # self.t = seq_len
        self.w_in = nn.Linear(self.d, self.n, bias=False)
        self.w_out = nn.Linear(self.n, self.d, bias=False)
        self.gru = nn.GRU(input_size=1, hidden_size=self.n, batch_first=True)

    def forward(self, x0s, seq_len):
        bz = x0s.shape[0]
        us = torch.zeros((bz, seq_len, 1))
        h0s = self.w_in(x0s).reshape((1, bz, self.n))
        hs, _ = self.gru(us, h0s)
        ys = self.w_out(hs)
        return ys

    def training_step(self, batch, _):
        x0s, xs = batch
        ys = self(x0s, xs.shape[1])
        loss = F.mse_loss(ys, xs)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "scheduler": scheduler}


def train(state_size, batch_size):

    dim = 2
    model = RNNDynamics(dim, state_size)
    datamodule = data.ODEDataModule(data.vanderpol, dim, 2, batch_size)

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    state_size = 64
    batch_size = 32
    train(state_size, batch_size)
