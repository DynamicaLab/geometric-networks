import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import torch
from torch import nn

plt.rcParams['font.size'] = 18


class RNN(nn.Module):

    def __init__(self, N, coordinates, tau=5, g=1.5, alpha=5, beta=0.5, rule='gaussian', sigma=1, h=0.05,
                 device='cuda:0'):

        self.device = device

        self._N = N
        self.tau = torch.tensor(tau).float().to(device)
        self.g = torch.tensor(g).float().to(device)
        self.alpha = torch.tensor(alpha).float().to(device)
        self.beta = torch.tensor(beta).float().to(device)

        self.coordinates = coordinates

        self.signs = np.sign(np.random.uniform(-1, 1, (N,)))
        W = self.signs * np.abs(self.generate_random_network(N, coordinates, rule=rule, sigma=sigma))
        self.W = torch.tensor(W).float().to(device)

        self.x_0 = torch.from_numpy(np.random.uniform(-1, 1, (N, 1))).float().to(device)
        self.r_0 = self.phi(self.x_0)
        self.X = [self.x_0]
        self.R = [self.r_0]

    def forward(self, T):
        self.r_l, self.x_l = self.R[-1], self.X[-1]
        for t in range(T):  # Euler integration scheme

            r = self.r_l
            x = self.x_l
            dx = (1 / self.tau) * (-x + self.g * torch.matmul(self.W, r))
            self.x_l = x + dx
            self.r_l = self.phi(self.x_l)

            self.X.append(self.x_l)
            self.R.append(self.r_l)

        return torch.concatenate(self.R, axis=1)

    def phi(self, x):
        return torch.tanh(x)
        # return 1 / (1 + torch.exp(-self.alpha * (x - self.beta))) # Logistic function

    @staticmethod
    def generate_random_network(N, coordinates, rule='gaussian', sigma=1, h=0.1):
        A = np.zeros((N, N))
        distances = compute_distances(coordinates, coordinates)
        if rule == 'gaussian':
            probabilities = np.exp((-(distances ** 2)) / (2 * sigma ** 2))
        elif rule == 'exponential':
            probabilities = np.exp(-sigma * distances)
        elif rule == 'sphere':
            W = (distances <= h).astype('float')
        W[np.diag_indices(N)] = 0
        N_edges = np.sum(W > 0)
        rho = N_edges / (N * (N - 1))
        W[W != 0] = np.random.normal(0, np.sqrt(1 / (N * rho)), N_edges)
        return W


@njit
def compute_distances(coords1, coords2):
    """Computes distances between two sets of N-D coordinates. Returns 2D matrix of distances."""
    N1, N2 = coords1.shape[0], coords2.shape[0]
    distances = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            distances[i, j] = np.sqrt(np.sum((coords1[i] - coords2[j]) ** 2))
    return distances