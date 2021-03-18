"""
Run the regression tasks on 1d data, this will train the models
"""
import argparse
import os
import os.path as osp
import time

from data.datasets import SineData, ExpData, LinearData, DampOscilData
from models.neural_process import *
from torch.utils.data import DataLoader
from models.training import TimeNeuralProcessTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, choices=['np', 'ndp', 'nd2p', 'vndp', 'vnd2p'],
    default='ndp')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--data', type=str, choices=['sine', 'exp', 'oscil', 'linear'], default='sine')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--load', type=eval, choices=[True, False], default=False)
args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run():
    # set device
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make folder
    folder = osp.join('results/1d', args.data, args.model, args.exp_name)
    if not osp.exists(folder):
        os.makedirs(folder)

    # Create dataset
    print('Loading Data')
    if args.data == 'sine':
        dataset = SineData(amplitude_range=(-1., 1.), shift_range=(-.5, .5), num_samples=500)
        initial_x = -3.2
    elif args.data == 'exp':
        dataset = ExpData(amplitude_range=(-1., 1.), shift_range=(-.5, .5), num_samples=500)
        initial_x = -1.1
    elif args.data == 'linear':
        dataset = LinearData(grad_range=(-1., 1.), shift_range=(-.5, .5), num_samples=500)
        initial_x = -0.1
    else:
        dataset = DampOscilData(amplitude_range=(-1., 1.), shift_range=(-.5, .5), num_samples=500)
        initial_x = -0.1
    initial_x = torch.tensor(initial_x).view(1, 1, 1).to(device)

    # select the process
    y_dim = 1

    nprocess = None
    if args.model == 'np':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        nprocess = MlpNeuralProcess(y_dim, r_dim, z_dim, h_dim).to(device)
    elif args.model == 'ndp':
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder
        L_dim = 10
        nprocess = MlpNeuralODEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'nd2p':
        r_dim = 50  # Dimension of representation of context points
        z_dim = 50  # Dimension of sampled latent variable
        h_dim = 50  # Dimension of hidden layers in encoder and decoder
        L_dim = 14
        nprocess = MlpNeuralODE2Process(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'vndp':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        L_dim = 10
        nprocess = VanillaNeuralODEProcess(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)
    elif args.model == 'vnd2p':
        r_dim = 60  # Dimension of representation of context points
        z_dim = 60  # Dimension of sampled latent variable
        h_dim = 60  # Dimension of hidden layers in encoder and decoder
        L_dim = 14
        nprocess = VanillaNeuralODE2Process(y_dim, r_dim, z_dim, h_dim, L_dim, initial_x).to(device)

    # Load model from checkpoint.
    if args.load:
        nprocess = torch.load(osp.join(folder, 'trained_model.pth')).to(device)

    torch.save(nprocess, osp.join(folder, 'untrained_model.pth'))

    batch_size = 5
    test_set_size = 10
    context_range = (1, 10)
    extra_target_range = (0, 5)

    nparams = np.array([count_parameters(nprocess)])
    print('Parameters = ' + str(nparams))
    np.save(osp.join(folder, 'parameter_count.npy'), nparams)

    data_loader = DataLoader(dataset[:int(len(dataset) - test_set_size)],
        batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset[int(len(dataset) - test_set_size):],
        batch_size=test_set_size, shuffle=False)
    optimizer = torch.optim.RMSprop(nprocess.parameters(), lr=1e-3)
    np_trainer = TimeNeuralProcessTrainer(device, nprocess, optimizer,
        num_context_range=context_range, num_extra_target_range=extra_target_range)

    print('Training')

    start_time = time.time()
    np_trainer.train(data_loader, test_data_loader, args.epochs)
    end_time = time.time()

    print('Total time = ' + str(end_time - start_time))
    np.save(osp.join(folder, 'training_time.npy'), np.array([end_time - start_time]))
    np.save(osp.join(folder, 'loss_history.npy'), np.array(np_trainer.epoch_loss_history))
    np.save(osp.join(folder, 'test_mse_history.npy'), np.array(np_trainer.epoch_mse_history))
    np.save(osp.join(folder, 'test_logp_history.npy'), np.array(np_trainer.epoch_logp_history))
    torch.save(nprocess, osp.join(folder, 'trained_model.pth'))


if __name__ == "__main__":
    run()
