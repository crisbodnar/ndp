import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from models.utils import context_target_split as cts


def plot_img_regression_prediction(neural_process, data_loader, num_context, t_min, t_max,
                                   foldername, title, use_y0=True):
    batch = next(iter(data_loader))
    x, y = batch

    # Extract device from model.
    device = next(neural_process.parameters()).device
    x_context, y_context, _, _, y0 = cts(x[0:1], y[0:1], num_context, 0, use_y0=use_y0)

    # Create a set of target points corresponding to entire [t_min, t_max] range
    extrapolation = 5
    x_target = torch.linspace(t_min, t_max + extrapolation / 10, x.shape[1] + extrapolation)
    x_target = x_target.unsqueeze(1).unsqueeze(0).to(device)
    x_target_rounded = torch.round(x_target * 10 ** 3) / (10 ** 3)

    # plt.scatter(x_context.flatten(), y_context.flatten())

    neural_process.eval()
    assert not neural_process.training

    # Neural process returns distribution over y_target
    p_y_pred = neural_process(x_context.to(device), y_context.to(device), x_target,
        y0=y0.to(device))
    # Extract mean of distribution
    mu = p_y_pred.mean.detach().cpu()
    img_width = int(np.sqrt(mu.shape[2]))
    im = mu.detach().numpy()[0].reshape(mu.shape[1], img_width, img_width)

    plt.figure(figsize=(33, 4))
    # print(x_target.shape, y_target.shape)
    x_target_rounded = x_target_rounded.cpu()
    x_target = x_target.cpu()
    x_context = x_context.cpu()
    for i, x in enumerate(x_target_rounded[0]):
        plt.subplot(2, x_target.cpu().shape[1], i + 1)
        if x in x_context:
            index = np.where((x_context == x))[1][0]
            plt.imshow(y_context[0][index].view(img_width, img_width), cmap='gray')
        else:
            plt.imshow(np.zeros((img_width, img_width)), cmap='gray')
        plt.axis('off')

        plt.subplot(2, x_target.shape[1], i + x_target.shape[1] + 1)
        plt.imshow(im[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(osp.join(foldername, f'{title}.pdf'))