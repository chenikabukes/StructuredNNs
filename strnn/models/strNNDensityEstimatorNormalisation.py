import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from strnn.models.strNNBatchNorm import StrNNBatchNorm
from numpy.random import binomial
from strnn.models.strNNBatchNorm import MaskedLinear
from strnn.models.model_utils import NONLINEARITIES

SUPPORTED_DATA_TYPES = ['binary', 'gaussian']


class StrNNDensityEstimatorNormalisation(StrNNBatchNorm):
    def __init__(self,
                 nin: int,
                 hidden_sizes: tuple[int, ...],
                 nout: int,
                 opt_type: str = 'greedy',
                 opt_args: dict = {'var_penalty_weight': 0.0},
                 precomputed_masks: np.ndarray | None = None,
                 adjacency: np.ndarray | None = None,
                 activation: str = 'relu',
                 data_type: str = 'binary'
                 ):
        assert data_type in SUPPORTED_DATA_TYPES
        self.data_type = data_type

        super().__init__(
            nin, hidden_sizes, nout, opt_type, opt_args,
            precomputed_masks, adjacency, activation
        )

        # Adding batch/layer normalization layers directly in the net_list
        new_net_list = nn.ModuleList()
        for layer in self.net_list:
            new_net_list.append(layer)
            if isinstance(layer, MaskedLinear) and layer.out_features != nout:
                # new_net_list.append(nn.BatchNorm1d(layer.out_features))
                new_net_list.append(nn.LayerNorm(layer.out_features))
                new_net_list.append(NONLINEARITIES[activation])
        self.net_list = new_net_list
        # Check if the last item is an instance of the activation function class
        if isinstance(self.net_list[-1], NONLINEARITIES[activation].__class__):
            self.net_list = self.net_list[:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net_list:
            x = layer(x)
        return x

    def compute_LL(self, x, x_hat):
        """
        Compute negative log likelihood given input x and reconstructed x_hat
        """
        mu, log_sigma = x_hat[:, :self.nin], x_hat[:, self.nin:]
        z = (x - mu) * torch.exp(-log_sigma)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = - log_sigma.sum(1) + log_prob_gauss

        return ll, z

    def get_preds_loss(self, batch):
        x = batch
        x_hat = self(x)
        assert self.data_type in SUPPORTED_DATA_TYPES

        if self.data_type == 'binary':
            # Evaluate the binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(
                x_hat, x, reduction='sum'
            ) / len(x)
        else:
            # Assume data is Gaussian if not binary
            loss = - self.compute_LL(x, x_hat)[0].sum() / len(x)

        return x_hat, loss

    def generate_sample(self, x0):
        """
        Generate a data sample using trained model
        BINARY VERSION ONLY AT THE MOMENT!!!

        @param x0: value of first data dimension
        @return: generated sample
        """
        sample = torch.from_numpy(np.zeros(self.nin))
        sample[0] = x0

        for d in range(1, self.nin):
            out = self(sample.float())
            sig = torch.nn.Sigmoid()
            out = sig(out)
            p_d = out[d]
            x_d = binomial(1, p=p_d.detach().numpy())
            sample[d] = x_d
        return sample


if __name__ == '__main__':
    A = np.ones((6, 3))
    A = np.tril(A, -1)
    model = StrNNDensityEstimatorNormalisation(
        nin=3,
        hidden_sizes=(6,),
        nout=6,
        opt_type='Zuko',
        opt_args={'var_penalty_weight': 0.0},
        precomputed_masks=None,
        adjacency=A,
        activation='relu')
    print(model.A)
