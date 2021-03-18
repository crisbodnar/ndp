import torch
import numpy as np
import time
import os
import os.path as osp
import argparse

from torch.utils.data import DataLoader
from data.data_loading import load_dataset
from models.neural_process import ConvNeuralProcess, ConvNeuralODEProcess
from models.training import TimeNeuralProcessTrainer
from plotting.img_regression_plotting import plot_img_regression_prediction

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--latent_only', action='store_true', help='Decode only latent state (no z)')
parser.add_argument('--exclude_time', action='store_true', help='Exclude time from the ODE')
parser.add_argument('--data', type=str, choices=['RotMNIST', 'VaryRotMNIST'], default='RotMNIST')
parser.add_argument('--model', type=str, choices=['np', 'ndp'], default='ndp')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--load', type=eval, choices=[True, False], default=False)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_context', type=int, default=9, help='Context size')
parser.add_argument('--eval_num_context', type=int, default=5, help='Eval context size')
parser.add_argument('--num_extra_target', type=int, default=6,
    help='Maximum number of extra target points')
parser.add_argument('--r_dim', type=int, default=100, help='Dimension of the aggregated context')
parser.add_argument('--z_dim', type=int, default=50, help='Dim of the latent sampled variable')
parser.add_argument('--L_dim', type=int, default=25, help='Dimension of the latent ODE')
parser.add_argument('--h_dim', type=int, default=100, help='Dim of the hidden layers in the ODE')
parser.add_argument('--lr', type=float, default=1e-3, help="Model learning rate")
parser.add_argument('--use_y0', action='store_true', help="Whether to use initial y or not")
parser.add_argument('--min_save_epoch', type=int, default=20,
    help="Epoch from which to start saving the model.")
parser.add_argument('--use_all_targets', type=eval, choices=[True, False], default=True,
    help="Use all the points in the time-series as target.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    # Parse and print arguments.
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = str(time.time())
    print(args)

    # Set device.
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Make folder
    folder = osp.join('results', args.data, args.model, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    # Create dataset
    print('Loading Data')
    train, val, test = load_dataset(args.data)
    y_dim = train.y_dim
    t_min = train.t_min
    t_max = train.t_max
    initial_t = torch.tensor(t_min).view(1, 1, 1).to(device)

    if args.model == 'np':
        neuralprocess = ConvNeuralProcess(y_dim, args.r_dim, args.z_dim, args.h_dim).to(device)
    elif args.model == 'ndp':
        neuralprocess = ConvNeuralODEProcess(y_dim, args.r_dim, args.z_dim,
            args.h_dim, args.L_dim, initial_t, latent_only=args.latent_only,
            exclude_time=args.exclude_time, use_y0=args.use_y0).to(device)
    else:
        raise ValueError(f'Unsupported model {args.model}')

    if args.load:
        print('Loading model from existent path...')
        neuralprocess = torch.load(osp.join(folder, 'trained_model.pth')).to(device)

    train_data_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=args.lr)
    np_trainer = TimeNeuralProcessTrainer(device, neuralprocess, optimizer,
        num_context_range=(1, args.num_context), num_extra_target_range=(1, args.num_extra_target),
        use_y0=args.use_y0, use_all_targets=args.use_all_targets)

    start_time = time.time()

    best_nll = 1e15
    for epoch in range(args.epochs):
        train_epoch_loss = np_trainer.train_epoch(train_data_loader)
        val_mse, val_nll = np_trainer.eval_epoch(val_data_loader)

        print(f'Epoch: {epoch} | Train loss: {train_epoch_loss:.3f} | Validation MSE: {val_mse:.3f} '
              f'| Val NLL: {val_nll:.3f}')

        if val_nll < best_nll and epoch > args.min_save_epoch:
            print('New best validation obtained. Checkpointing model and plotting...')
            best_nll = val_nll

            plot_img_regression_prediction(neural_process=neuralprocess,
                data_loader=train_data_loader, num_context=args.num_context, t_min=t_min,
                t_max=t_max, foldername=folder, title='train_pred', use_y0=args.use_y0)
            plot_img_regression_prediction(neural_process=neuralprocess,
                data_loader=val_data_loader, num_context=args.num_context, t_min=t_min,
                t_max=t_max, foldername=folder, title='val_pred', use_y0=args.use_y0)

            torch.save(neuralprocess, osp.join(folder, 'trained_model.pth'))

    end_time = time.time()
    np.save(osp.join(folder, 'training_time.npy'), np.array([end_time - start_time]))
    np.save(osp.join(folder, 'loss_history.npy'), np.array(np_trainer.epoch_loss_history))
    np.save(osp.join(folder, 'test_mse_history.npy'), np.array(np_trainer.epoch_mse_history))
    np.save(osp.join(folder, 'test_logp_history.npy'), np.array(np_trainer.epoch_logp_history))

    print("==========================")
    print(f'Total time = {end_time - start_time}')
    print(f"Best MSE: {best_nll:.4f}")


if __name__ == "__main__":
    run()
