import numpy as np
import scipy.io as sio
import os
import torch

from torch.utils.data import Dataset


class RotMNISTNonUniform(Dataset):
    """Code adapted from https://github.com/cagatayyildiz/ODE2VAE"""

    def __init__(self, dt=0.1, split='train'):
        # Fix the random seed for this dataset
        np.random.seed(0)

        data = sio.loadmat(os.path.join("datasets", "rot_mnist", 'rot-mnist-3s.mat'))
        Xtr = np.squeeze(data['X'])
        Xtr = np.flip(Xtr, 1)
        idx = np.arange(Xtr.shape[0])
        np.random.shuffle(idx)
        Xtr = Xtr[idx, :, :]

        Ntr = 360
        Xval = Xtr[Ntr:Ntr + Ntr//10]
        Xtest = Xtr[0:Ntr]

        removed_angle = 3
        [N, T, D] = Xtest.shape
        self.y_dim = D

        print("Dataset len", len(Xtr))

        num_gaps = 5
        ttr = np.zeros([N, T - num_gaps])
        Xtr = np.zeros([N, T - num_gaps, D])
        for i in range(N):
            idx = np.arange(0, T)
            d = {removed_angle}
            while len(d) < num_gaps:
                d.add(np.random.randint(1, T))
            idx = np.delete(idx, list(d))
            Xtr[i, :, :] = Xtest[i, idx, :]
            ttr[i, :] = dt * idx

        tval = dt * np.tile(np.arange(0, T).reshape((1, -1)), [Xval.shape[0], 1])
        ttest = dt * np.tile(np.arange(0, T).reshape((1, -1)), [N, 1])

        ttr = np.expand_dims(ttr, axis=2)
        tval = np.expand_dims(tval, axis=2)
        ttest = np.expand_dims(ttest, axis=2)

        if split == 'train':
            self.data = self.get_split(ttr, Xtr)
        elif split == 'val':
            self.data = self.get_split(tval, Xval)
        elif split == 'test':
            self.data = self.get_split(ttest, Xtest)
        else:
            raise ValueError(f'Unsupported split {split}')

        self.t_min = 0
        self.t_max = np.max(ttest)
        self.num_samples = len(self.data)

    def get_split(self, T, Y):
        assert T.shape[:-1] == Y.shape[:-1]
        assert T.shape[-1] == 1
        assert Y.shape[-1] == self.y_dim

        data = []
        for i in range(len(T)):
            t = torch.tensor(T[i], dtype=torch.float32)
            y = torch.tensor(Y[i], dtype=torch.float32)
            data.append((t, y))

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
