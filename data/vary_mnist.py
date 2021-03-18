import numpy as np
import os
import torch

from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class VaryRotMNIST(Dataset):
    """Code adapted from https://github.com/cagatayyildiz/ODE2VAE"""

    def __init__(self, split='train', num_points=5000, num_velocity_perms=3, dt=0.1):
        """
        Generates a dataset of rotated MNIST digits at different velocities and initial rotations.

        Parameters
        ----------
        time_points:		number of rotation frames
        num_points:			number of datapoints
        num_velocity_perms: number of velocity permutations per datapoint
        """
        # Fix the random seed for this dataset
        np.random.seed(0)

        savepath = os.path.join('datasets', 'rot_mnist', 'vary_mnist.pt')
        if os.path.exists(savepath):
            with open(savepath, 'rb') as f:
                Xtr = torch.load(f)
        else:
            Xtr = self.generate_data(16, num_points, num_velocity_perms, dt)
            with open(savepath, 'wb') as f:
                torch.save(Xtr, f)

        Xtr = np.flip(Xtr, 1)
        idx = np.arange(Xtr.shape[0])
        np.random.shuffle(idx)
        Xtr = Xtr[idx, :, :]
        Xtr_full = Xtr

        Ntr = 1000
        Xval = Xtr[Ntr:Ntr + Ntr//10]
        Xtest = Xtr[Ntr + Ntr//10:Ntr + 3*Ntr//10]
        Xtr = Xtr[0:Ntr]

        print("Dataset length", len(Xtr))
        print(f"Min: {np.min(Xtr_full)}, Max: {np.max(Xtr_full)}")

        [N, T, D] = Xtr.shape
        self.y_dim = D

        num_gaps = 5
        ttr = np.zeros([N, T - num_gaps])
        Xtr = np.zeros([N, T - num_gaps, D])
        for i in range(N):
            idx = np.arange(0, T)
            d = set()
            while len(d) < num_gaps:
                d.add(np.random.randint(1, T))
            idx = np.delete(idx, list(d))
            Xtr[i, :, :] = Xtr_full[i, idx, :]
            ttr[i, :] = dt * idx

        tval = dt * np.tile(np.arange(0, T).reshape((1, -1)), [Xval.shape[0], 1])
        ttest = dt * np.tile(np.arange(0, T).reshape((1, -1)), [Xtest.shape[0], 1])

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

    def generate_data(self, time_points, num_points, num_velocity_perms, dt):
        self.t = (torch.arange(time_points) / torch.tensor(10.)).view(-1, 1)

        data_train = datasets.MNIST(os.path.join('./datasets', 'mnist'),
                                    download=True,
                                    train=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()]))

        dataset = list()
        count = 0
        for i, (input, label) in enumerate(data_train):
            if label != 3:
                continue
            if num_points is not None and count >= num_points:
                break
            count += 1

            for velocity_perm in range(num_velocity_perms):
                datapoint = list()
                total_rotation = np.random.uniform(300, 420, 1)[0]
                start_rotation = total_rotation * np.random.choice(range(time_points)) / time_points

                for t in range(time_points):
                    im_rotate = ndimage.rotate(np.asarray(input[0]), start_rotation + total_rotation * t / time_points,
                                               reshape=False)
                    im_rotate = np.asarray(im_rotate).reshape(-1)
                    im_rotate[im_rotate < 0] = 0
                    im_rotate[im_rotate > 1] = 1
                    datapoint.append(im_rotate)

                dataset.append(datapoint)
        # np.random.shuffle(dataset) # Already shuffling after
        return np.stack(dataset)


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
