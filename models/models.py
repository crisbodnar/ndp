import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
import numpy as np

from torch.distributions import Bernoulli, Normal


class ConvEncoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i where y_i is a
    28*28 length vector representation of an image.

    Parameters
    ----------
    y_dim : int
        Dimension of y values.
    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, y_dim, r_dim):
        super(ConvEncoder, self).__init__()
        n_filt = 16
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.img_width = int(np.sqrt(y_dim))

        self.conv1 = nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2, 2))
        self.conv2 = nn.Conv2d(n_filt, n_filt * 2, kernel_size=5, stride=2, padding=(2, 2))
        self.conv3 = nn.Conv2d(n_filt * 2, n_filt * 4, kernel_size=5, stride=2, padding=(2, 2))
        self.conv4 = nn.Conv2d(n_filt * 4, n_filt * 8, kernel_size=5, stride=2, padding=(2, 2))
        self.fc1 = nn.Linear(128 * (2**2) + 128, r_dim)
        self.bn1 = nn.BatchNorm2d(n_filt)
        self.bn2 = nn.BatchNorm2d(n_filt * 2)
        self.bn3 = nn.BatchNorm2d(n_filt * 4)
        self.bn4 = nn.BatchNorm2d(n_filt * 8)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        """
        y = y.view(y.shape[0], 1, self.img_width, self.img_width)
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = y.view(y.size()[0], -1)  # Flatten

        input = torch.cat((x.repeat(1, 128), y), dim=1)
        output = self.fc1(input)
        return output


class Y0ConvEncoder(nn.Module):
    """Maps y0 to L_r

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, y_dim, r_dim):
        super(Y0ConvEncoder, self).__init__()
        n_filt = 8
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.img_width = int(np.sqrt(y_dim))

        self.conv1 = nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2, 2))
        self.conv2 = nn.Conv2d(n_filt, n_filt * 2, kernel_size=5, stride=2, padding=(2, 2))
        self.conv3 = nn.Conv2d(n_filt * 2, n_filt * 4, kernel_size=5, stride=2, padding=(2, 2))
        self.conv4 = nn.Conv2d(n_filt * 4, n_filt * 8, kernel_size=5, stride=2, padding=(2, 2))

        self.bn1 = nn.BatchNorm2d(n_filt)
        self.bn2 = nn.BatchNorm2d(n_filt * 2)
        self.bn3 = nn.BatchNorm2d(n_filt * 4)
        self.bn4 = nn.BatchNorm2d(n_filt * 8)

        self.fc1 = nn.Linear(n_filt * 8 * (2**2), r_dim)

    def forward(self, y0):
        """
        y0 : torch.Tensor
            Shape (batch_size, y_dim)
        """
        y = y0.view(y0.shape[0], 1, self.img_width, self.img_width)
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = y.view(y.size()[0], -1)  # Flatten
        output = self.fc1(y)
        return output


class SingleContextNDPEncoder(nn.Module):
    """Use Context and the same conv network to infer both L(0) and D"""
    def __init__(self, context_encoder):
        super(SingleContextNDPEncoder, self).__init__()
        self.context_encoder = context_encoder

    def forward(self, x, y, _):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        """
        output = self.context_encoder(x, y)
        return output, output


class Y0ContextNDPEncoder(nn.Module):
    """Use y0 to infer L(0) and Context to infer D"""
    def __init__(self, y0_encoder, context_encoder):
        super(Y0ContextNDPEncoder, self).__init__()
        self.y0_encoder = y0_encoder
        self.context_encoder = context_encoder

    def forward(self, x, y, y0):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim)
        """
        L_output = self.y0_encoder(y0)
        D_output = self.context_encoder(x, y)
        return L_output, D_output


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, x_dim)

        y : torch.Tensor
            Shape (batch_size, y_dim)
        """
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class NPMlpDecoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(NPMlpDecoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class ConvDecoderNet(nn.Module):
    """Network doing the convolutional decoding in the NP and NDP decoders."""

    def __init__(self, y_dim):
        super(ConvDecoderNet, self).__init__()

        img_width = int(np.sqrt(y_dim))
        self.feat = 3 if img_width == 28 else 4
        padding1 = (0, 0) if img_width == 28 else (2, 2)
        padding2 = (0, 0) if img_width == 28 else (1, 1)
        stride = 1 if img_width == 28 else 2

        self.decode_conv1 = nn.ConvTranspose2d(8, 128, kernel_size=3, stride=stride,
            padding=padding1)
        self.decode_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=padding2)
        self.decode_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=(1, 1),
            output_padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.hidden_to_mu = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=(2, 2))

    def forward(self, x):
        x = x.view(x.shape[0], 8, self.feat, self.feat)
        x = F.relu(self.bn1(self.decode_conv1(x)))
        x = F.relu(self.bn2(self.decode_conv2(x)))
        x = F.relu(self.bn3(self.decode_conv3(x)))
        x = self.hidden_to_mu(x)
        return x


class NPConvDecoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, conv_decoder_net):
        super(NPConvDecoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        self.feat = 3 if int(np.sqrt(y_dim)) == 28 else 4
        self.decode_fc = nn.Linear(x_dim + z_dim, self.feat * self.feat * 8)
        self.conv_decoder = conv_decoder_net

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        inp = torch.cat((x_flat, z_flat), dim=1)
        inp = self.decode_fc(inp)

        mu = self.conv_decoder(inp)
        mu = mu.view(batch_size, num_points, self.y_dim)
        return Bernoulli(logits=mu)


class AbstractODEDecoder(nn.Module):
    """
    An Abstract Decoder using a Neural ODE. Child classes must implement decode_latent.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(AbstractODEDecoder, self).__init__()
        # The input is always time.
        assert x_dim == 1

        self.exclude_time = exclude_time
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim
        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]
        # z = [L0, z_] so dim([L, z_, x]) = dim(z)+1
        self.latent_odefunc = nn.Sequential(*ode_layers)

        self.decode_layers = [nn.Linear(x_dim + z_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True)]
        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)

        self.initial_t = initial_t
        self.nfe = 0

    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        dL = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dL, dz_), dim=1)

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        raise NotImplementedError('The decoding of the latent ODE state is not implemented')

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, 1)
        z : torch.Tensor
            Shape (batch_size, z_dim)
        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        # Append the initial time to the set of supplied times.
        x0 = self.initial_t.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        # ind specifies where each element in x ended up in times.
        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)
        # Remove the initial position index since we don't care about it.
        ind = ind[:, 1:, :]

        # Integrate forward from the batch of initial positions z.
        v = odeint(self.integrate_ode, z, times, method='dopri5')

        # Make shape (batch_size, unique_times, z_dim).
        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        # Extract the relevant (latent, time) pairs for each batch.
        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)

        return self.decode_latent(x, z, latent)


class MlpNormalODEDecoder(AbstractODEDecoder):
    """
    Maps target times x_target (which we call x for consistency with NPs)
    and samples z (encoding information about the context points)
    to predictions y_target. The decoder is an ODEsolve, using torchdiffeq.
    This version contains no control.
    Models inheriting from MlpNormalODEDecoder *must* either set self.xlz_to_hidden
    in constructor or override decode_latent(). Optionally, integrate_ode
    and forward can also be overridden.

    Parameters
    ----------
    x_dim : int
        Dimension of x values. Currently only works for dimension of 1.
    z_dim : int
        Dimension of latent variable z. Contains [L0, z_].

    h_dim : int
        Dimension of hidden layer in odefunc.
    y_dim : int
        Dimension of y values.

    L_dim : int
        Dimension of latent state L.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None):
        super(MlpNormalODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim)

        self.x_dim = x_dim  # must be 1
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        # Input is concatenation of z with every row of x
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)

        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return Normal(mu, sigma)


# Includes batching, now includes a latent state to go through MLP to get mu/sigma
class MlpSonodeDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(MlpSonodeDecoder, self).__init__(x_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
            L_out_dim=int(L_dim / 2))

    # x needs to be written as t here to use torchdiffeq
    # v, and therefore L and z_ must be vectors as a torch tensor, so they can concatentate
    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        Lv = v[:, int(self.L_dim / 2):self.L_dim]
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        time = t.view(1, 1).repeat(batch_size, 1)
        vt = torch.cat((v, time), dim=1)
        dLx = Lv
        dLv = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dLx, dLv, dz_), dim=1)


class VanillaODEDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(VanillaODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim, initial_t)

        self.latent_to_mu = nn.Linear(L_dim, y_dim)
        self.latent_to_sigma = nn.Linear(L_dim, y_dim)

    def decode_latent(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        latent_flat = latent.view(batch_size * num_points, -1)
        mu = self.latent_to_mu(latent_flat)
        pre_sigma = self.latent_to_sigma(latent_flat)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class VanillaSONODEDecoder(MlpNormalODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t):
        super(VanillaSONODEDecoder, self).__init__(
            x_dim, z_dim, h_dim, y_dim, L_dim,
            initial_t, L_out_dim=int(L_dim / 2))

        self.latent_to_mu = nn.Linear(L_dim, y_dim)
        self.latent_to_sigma = nn.Linear(L_dim, y_dim)

    # x needs to be written as t here to use torchdiffeq
    # v, and therefore L and z_ must be vectors as a torch tensor, so they can concatentate
    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        Lv = v[:, int(self.L_dim / 2):self.L_dim]
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        time = t.view(1, 1).repeat(batch_size, 1)
        vt = torch.cat((v, time), dim=1)
        dLx = Lv
        dLv = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)
        return torch.cat((dLx, dLv, dz_), dim=1)

    def decode_latent(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        latent_flat = latent.view(batch_size * num_points, -1)
        mu = self.latent_to_mu(latent_flat)
        pre_sigma = self.latent_to_sigma(latent_flat)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class ConvODEDecoder(AbstractODEDecoder):
    """
    MlpNormalODEDecoder with a transposed convolutional decoder.
    """

    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, conv_decoder_net,
                 latent_only=False, exclude_time=False):
        super(ConvODEDecoder, self).__init__(x_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
            exclude_time=exclude_time)
        self.latent_only = latent_only
        self.feat = 3 if int(np.sqrt(y_dim)) == 28 else 4

        if self.latent_only:
            self.decode_fc = nn.Linear(L_dim, self.feat * self.feat * 8)
        else:
            self.decode_fc = nn.Linear(x_dim + z_dim, self.feat * self.feat * 8)

        self.conv_decoder = conv_decoder_net

    def decode_latent(self, x, z, latent):
        batch_size, num_points, _ = x.size()
        # compute sigma using mlp (t, L(t), z_)
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        # Input is concatenation of z with every row of x

        input = latent_flat if self.latent_only else torch.cat((x_flat, latent_flat, z_flat), dim=1)
        input = self.decode_fc(input)

        mu = self.conv_decoder(input)
        mu = mu.view(batch_size, num_points, self.y_dim)
        return Bernoulli(logits=mu)
