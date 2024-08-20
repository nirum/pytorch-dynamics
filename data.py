import numpy as np
from scipy.integrate import solve_ivp

from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl


def lorenz(_, x, sigma=10., beta=(8/3), rho=28.):
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def vanderpol(_, x, mu=2.0):
    return np.array([
        x[1],
        mu * (1 - x[0] ** 2) * x[1] - x[0],
    ])
    

class ODEDataset(IterableDataset):

    def __init__(self, dxdt, dim, num_examples, x0_lim, dt=1e-2, seq_len=100, sigma=1e-2, seed=1234):
        self.dxdt = dxdt
        self.dim = dim
        self.num_examples = num_examples
        self.dt = dt
        self.seq_len = seq_len
        self.t = np.linspace(0, dt * (seq_len - 1), seq_len)
        self.x0_lim = x0_lim
        self.sigma = sigma
        self.rs = np.random.RandomState(seed)

    def __iter__(self):
        x0s = np.random.rand(self.num_examples, self.dim) * 2 * self.x0_lim - self.x0_lim
        for x0 in x0s:
            res = solve_ivp(self.dxdt, [self.t[0], self.t[-1]], x0, t_eval=self.t)
            data = res.y + self.sigma * self.rs.randn(*res.y.shape)
            yield (x0.astype(np.float32), data.T.astype(np.float32))


class ODEDataModule(pl.LightningDataModule):
    def __init__(self, dxdt, dim, x0_lim, batch_size, seq_len, dt, sigma=0.01):
        super().__init__()
        self.dxdt = dxdt
        self.dim = dim
        self.x0_lim = x0_lim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dt = dt
        self.sigma = sigma

    def setup(self, stage: str):
        del stage
        self.train_dset = ODEDataset(self.dxdt, self.dim, self.batch_size * 100, self.x0_lim, dt=self.dt, seq_len=self.seq_len, sigma=self.sigma)
        self.val_dset = ODEDataset(self.dxdt, self.dim, self.batch_size * 10, self.x0_lim, dt=self.dt, seq_len=self.seq_len, sigma=self.sigma)
        self.test_dset = ODEDataset(self.dxdt, self.dim, self.batch_size * 10, self.x0_lim, dt=self.dt, seq_len=self.seq_len, sigma=self.sigma)

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size) #, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size) #, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size) #, shuffle=False)
