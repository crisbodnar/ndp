import torch
import numpy as np
import time
import os
import os.path as osp
import argparse
import seaborn as sns

sns.set()

from torch.utils.data import DataLoader
from data.data_loading import load_dataset
from models.utils import context_target_split as cts

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name1', type=str, default=None)
parser.add_argument('--exp_name2', type=str, default=None)
parser.add_argument('--exp_name', type=str, default=str(time.time()))

parser.add_argument('--data', type=str, choices=['RotMNIST', 'VaryRotMNIST'],
    default='RotMNISTNonUniform')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num_context', type=int, default=5, help='Context size')
parser.add_argument('--eval_num_context', type=int, default=5, help='Eval context size')
parser.add_argument('--num_extra_target', type=int, default=6,
    help='Maximum number of extra target points')
parser.add_argument('--r_dim', type=int, default=50, help='Dimension of the aggregated context')
parser.add_argument('--z_dim', type=int, default=70, help='Dim of the latent sampled variable')
parser.add_argument('--L_dim', type=int, default=25, help='Dimension of the latent ODE')
parser.add_argument('--h_dim', type=int, default=100,
    help='Dimension of the hidden layers in the encoder and the decoder')
parser.add_argument('--lr', type=float, default=1e-3, help="Model learning rate")


def plot_comparison(neuralprocess1, neuralprocess2, data_loader, device, name, context_size=5):
    preds = list()
    img_width = 28
    batch = next(iter(data_loader))
    # Create context and target points and apply neural process
    x, y = batch
    # locations = torch.arange(x.size(1))
    x_context, y_context, x_target, y_target, y0 = cts(x, y, context_size, x.size(1) - context_size)
    y0 = y0.to(device)
    t_min = x_target.min()
    t_max = x_target.max()

    extrapolation = 5

    x_target = torch.linspace(t_min, t_max + extrapolation / 10, x.shape[1] + extrapolation)
    x_target = x_target.unsqueeze(1).unsqueeze(0).to(device).repeat(data_loader.batch_size, 1, 1)

    x_context = x_context.to(device)
    y_context = y_context.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)

    p_y_pred = neuralprocess1(x_context, y_context, x_target, y, y0)
    pred1 = p_y_pred.mean.detach().cpu()
    p_y_pred = neuralprocess2(x_context, y_context, x_target, y, y0)
    pred2 = p_y_pred.mean.detach().cpu()
    im1 = pred1.numpy().reshape(data_loader.batch_size, pred1.shape[1], img_width, img_width)
    im2 = pred2.numpy().reshape(data_loader.batch_size, pred2.shape[1], img_width, img_width)
    images = [im2, im1]

    for batch_index in range(0):
        x_target = x_target.cpu()
        x_context = x_context.cpu()
        x_target_rounded = torch.round(x_target * 10 ** 3) / (10 ** 3)
        x_target_rounded = x_target_rounded.cpu()
        fig = plt.figure(figsize=(20, 4.3))
        gs = gridspec.GridSpec(nrows=4, ncols=x_target.shape[1], width_ratios=[1] * x_target.shape[1],
                               wspace=0.04, hspace=0.04, top=0.86, bottom=0.05, left=0.07, right=0.99)
        # print(x_target.shape, y_target.shape)
        for i, x in enumerate(x_target_rounded[0]):
            ax0 = plt.subplot(gs[0, i])
            if i < y[0].shape[0]:
                ax0.imshow(y[batch_index][i].view(img_width, img_width), cmap='gray')
            else:
                ax0.imshow(np.zeros((img_width, img_width)), cmap='gray')
            ax0.axis('off')

            ax1 = plt.subplot(gs[1, i])
            if x in x_context:
                index = np.where((x_context == x))[1][0]
                ax1.imshow(y_context[batch_index][index].view(img_width, img_width), cmap='gray')
            else:
                ax1.imshow(np.zeros((img_width, img_width)), cmap='gray')
            ax1.axis('off')

            for k in range(2):
                ax = plt.subplot(gs[k+2, i])
                ax.imshow(images[k][batch_index][i], cmap='gray')
                ax.axis('off')

        plt.gcf().text(0.01, 0.75, 'Target', color='black', size=25)
        plt.gcf().text(0, 0.55, 'Context', color='black', size=25)
        plt.gcf().text(0.035, 0.35, 'NP', size=25)
        plt.gcf().text(0.025, 0.14, 'NDP', size=25)
        plt.gcf().text(0.8, 0.92, 'extrapolation', size=25)
        plt.annotate('',
                     xy=(0.99, 0.89),  # theta, radius
                     xytext=(0.77, 0.89),  # fraction, fraction
                     xycoords='figure fraction',
                     textcoords='figure fraction',
                     arrowprops=dict(color='black', linewidth=1.5, arrowstyle='->'),
                     horizontalalignment='left',
                     verticalalignment='bottom')

        # plt.tight_layout()
        plt.savefig(osp.join('./', f'{name}-{batch_index}_comparison.pdf'))

    # Generate styles image and calculate pixel-MSE
    print('Calculating Pixel MSE and Styles')
    numstyles = 12
    fig = plt.figure(figsize=(13, 2.4))
    gs = gridspec.GridSpec(nrows=2, ncols=numstyles, width_ratios=[1]*numstyles,
                            wspace=0.04, hspace=0.04, top=0.95, bottom=0.05, left=0.01, right=0.99)
    ndp_images = images[1]

    loc = [0, 6, 11, 8, 6, 10, 4, 2, 7, 3, 0, 0]
    for i in range(numstyles):
        ax0 = plt.subplot(gs[0, i])
        #[i, 0]
        ax0.imshow(y[i, loc[i]].numpy().reshape(img_width, img_width), cmap='gray')
        plt.axis('off')
        ax1 = plt.subplot(gs[1, i])
        ax1.imshow(ndp_images[i, loc[i]], cmap='gray')
        ax1.axis('off')

    plt.savefig(f'{name}styles.pdf')


def run():
    # Parse and print arguments.
    args = parser.parse_args()
    print(args)

    # Set device.
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Make folders
    folder1 = osp.join('results', args.data, 'cndp', args.exp_name1)
    if not osp.exists(folder1):
        raise ValueError(f'Experiment {folder1} does not exist')
    folder2 = osp.join('results', args.data, 'cnp', args.exp_name2)
    if not osp.exists(folder2):
        raise ValueError(f'Experiment {folder2} does not exist')

    # Create dataset
    print('Loading Data')
    train, val, test = load_dataset(args.data)

    print('Loading model from existent path...')
    neuralprocess1 = torch.load(osp.join(folder1, 'trained_model.pth'), map_location=device)
    neuralprocess1.eval()

    neuralprocess2 = torch.load(osp.join(folder2, 'trained_model.pth'), map_location=device)
    neuralprocess2.eval()

    test_data_loader = DataLoader(test[:20], batch_size=20, shuffle=False)
    plot_comparison(neuralprocess1, neuralprocess2, test_data_loader, device, args.exp_name)


if __name__ == "__main__":
    run()
