from data.mnist_nonuniform import RotMNISTNonUniform
from data.vary_mnist import VaryRotMNIST


def load_dataset(name):
    if name == 'RotMNIST':
        train = RotMNISTNonUniform(split='train')
        val = RotMNISTNonUniform(split='val')
        test = RotMNISTNonUniform(split='test')
        return train, val, test
    elif name == 'VaryRotMNIST':
        train = VaryRotMNIST(split='train')
        val = VaryRotMNIST(split='val')
        test = VaryRotMNIST(split='test')
        return train, val, test
    else:
        raise ValueError(f'Unsupported dataset {name}')
