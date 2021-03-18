import torch

from tqdm import tqdm
from typing import Tuple
from random import randint
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli, Normal
from models.utils import context_target_split as cts
from models.neural_process import TimeNeuralProcess
from torch.utils.data import DataLoader


class TimeNeuralProcessTrainer:
    """
    Class to handle training of Neural Processes.
    Code adapted from https://github.com/EmilienDupont/neural-processes

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.TimeNeuralProcess, neural_process.NeuralODEProcess instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    """
    def __init__(self,
                 device: torch.device,
                 neural_process: TimeNeuralProcess,
                 optimizer: torch.optim.Optimizer,
                 num_context_range: Tuple[int, int],
                 num_extra_target_range: Tuple[int, int],
                 max_context=None,
                 use_all_targets=False,
                 use_y0=True):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.max_context = max_context
        self.use_all_targets = use_all_targets
        self.use_y0 = use_y0

        self.epoch_loss_history = []
        self.epoch_nfe_history = []
        self.epoch_mse_history = []
        self.epoch_logp_history = []

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader, epochs: int):
        """
        Trains Neural (ODE) Process.

        Parameters
        ----------
        train_data_loader : Data loader to use for training
        val_data_loader: Data loader to use for validation
        epochs: Number of epochs to train for
        """
        self.neural_process.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch}')
            epoch_loss = self.train_epoch(train_data_loader)
            # some bit about the self.epoch_nfe_history.append if we want to track nfe in training
            self.epoch_loss_history.append(epoch_loss)
            self.eval_epoch(val_data_loader)

    def train_epoch(self, data_loader):
        epoch_loss = 0.
        self.neural_process.train()
        for i, data in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()

            # Extract data
            x, y = data
            points = x.size(1)

            # Sample number of context and target points
            num_context = randint(*self.num_context_range)
            num_extra_target = randint(*self.num_extra_target_range)
            if self.use_all_targets:
                num_extra_target = points - num_context

            # Create context and target points and apply neural process
            x_context, y_context, x_target, y_target, y0 = (
                cts(x, y, num_context, num_extra_target, use_y0=self.use_y0))
            y0 = y0.to(self.device)
            x_context = x_context.to(self.device)
            y_context = y_context.to(self.device)
            x_target = x_target.to(self.device)
            y_target = y_target.to(self.device)

            p_y_pred, q_target, q_context = (
                self.neural_process(x_context, y_context, x_target, y_target, y0))
            loss = self._loss(p_y_pred, y_target, q_target, q_context)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.cpu().item()

        return epoch_loss / len(data_loader)

    def eval_epoch(self, data_loader, context_size=None):
        """Runs in eval mode on the given data loader and uses the whole time series as target."""
        epoch_mse = 0
        epoch_nll = 0
        if context_size is None:
            context_size = randint(*self.num_context_range)

        self.neural_process.eval()
        for i, data in enumerate( tqdm(data_loader)):
            with torch.no_grad():
                x, y = data
                x_context, y_context, _, _, y0 = cts(x, y, context_size, 0, use_y0=self.use_y0)

                y0 = y0.to(self.device)
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)

                # Use the whole time series as target.
                x_target = x.to(self.device)
                y_target = y.to(self.device)
                p_y_pred = self.neural_process(x_context, y_context, x_target, y_target, y0)

                nll = self._loss(p_y_pred, y_target)
                epoch_nll += nll.cpu().item()

                mse = ((y_target-p_y_pred.mean)**2).mean()
                epoch_mse += mse.item()

        epoch_mse = epoch_mse / len(data_loader)
        epoch_nll = epoch_nll / len(data_loader)
        self.epoch_mse_history.append(epoch_mse)
        self.epoch_logp_history.append(epoch_nll)

        return epoch_mse, epoch_nll

    def _loss(self, p_y_pred, y_target, q_target=None, q_context=None):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        if isinstance(p_y_pred, Bernoulli):
            # Pixels might be in (0, 1), but we still treat them as binary
            # so this is a bit of a hack. This is needed because pytorch checks the argument
            # to log_prob is in the support of the Bernoulli distribution (i.e. it is 0 or 1).
            pred = p_y_pred.logits
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            nll = loss(pred, y_target).mean(dim=0).sum()
        else:
            nll = -p_y_pred.log_prob(y_target).mean(dim=0).sum()

        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        if q_target is None and q_context is None:
            return nll

        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return nll + kl

