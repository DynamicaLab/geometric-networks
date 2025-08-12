import copy
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
import statsmodels.sandbox.stats.multicomp as mc
from brainspace.gradient.embedding import diffusion_mapping


# ----------------------------------------------------------------------------------------------------------------------
# Classes for simulated dynamics in various geometries
# ----------------------------------------------------------------------------------------------------------------------

class Geometry:

    def __init__(self, vertices, eigenmodes):
        order = np.argsort(vertices[:, 2])
        self.vertices = vertices[order]
        self.eigenmodes = eigenmodes[order]
        if self.eigenmodes.shape[0] == self.vertices.shape[0]:
            self.eigenmodes = self.eigenmodes.T
        self.random_ids = None

    def subsample_coordinates(self, N_nodes):
        self.random_ids = np.sort(np.random.choice(self.vertices.shape[0], N_nodes))
        return self.vertices[self.random_ids]

    def subsample_eigenmodes(self):
        return self.eigenmodes[:, self.random_ids]


class Simulator:

    def __init__(self, geometry, dynamics, params, device='cuda:0'):

        self.device = device

        self.geometry = geometry
        self.dynamics_class = dynamics
        self.dynamics_params = params
        self.N_neurons = params['N_neurons']
        self.dynamics = None
        self.coordinates = None
        self.timeseries = None
        self.reinitialize_dynamics()
        self.animation = None

    def reinitialize_dynamics(self, new_coordinates=True, W_0=None):
        if self.coordinates is None:
            self.coordinates = self.geometry.subsample_coordinates(self.N_neurons)
            self.coordinates -= np.mean(self.coordinates, axis=0)
        if new_coordinates:
            self.coordinates = self.geometry.subsample_coordinates(self.N_neurons)
            self.coordinates -= np.mean(self.coordinates, axis=0)
        self.dynamics = self.dynamics_class(self.coordinates, self.dynamics_params, device=self.device)
        if W_0 is not None: # For edge swapping, allows the introduction of an externally generated W matrix
            W = self.dynamics.W.detach().cpu().numpy()
            weights = W[W != 0]
            W_0[W_0 != 0] = weights[:np.sum(W_0 != 0)]
            self.dynamics.W = torch.tensor(W_0).float().to(self.device)

    def compute_average_correlations(self, n_iters=100, T=500, verbose=True, calcium=False, tau_calcium=10,
                                     smoothing=False, sigma=0.05, W_0=None):
        N = self.dynamics.coordinates.shape[0]
        C = np.zeros((N, N))
        for i in progress_bar(range(n_iters), verbose=verbose):
            self.reinitialize_dynamics(new_coordinates=False, W_0=W_0)
            R = self.dynamics.integrate(T)
            if calcium:
                R = gcampify(R, tau=tau_calcium)
            if smoothing:
                R = spatial_smoothing(self.torch_to_numpy(R), self.coordinates, sigma=sigma)
            if type(R) == torch.Tensor:
                c = torch.corrcoef(R).detach().cpu().numpy()
            else:
                c = np.corrcoef(R)
            c[np.isnan(c)] = 0
            C = (np.abs(C) * (i / (1 + i))) + (np.abs(c) * (1 / (i + 1)))
        C[np.diag_indices(C.shape[0])] = 0
        return C

    def compute_geometric_mapping(self, C, N_modes=50, alpha=0.5, return_gradients=False):
        C[np.diag_indices(C.shape[0])] = 0
        modes_functional, _ = diffusion_mapping(C, n_components=N_modes, alpha=alpha)
        modes_functional = modes_functional.T
        modes_geometric = self.geometry.subsample_eigenmodes()
        modes_geometric = modes_geometric[1:N_modes + 1, :]
        mode_similarity, mapping = compute_mode_similarity_matrix(modes_geometric, modes_functional, return_mapping=True)
        if return_gradients:
            return mode_similarity, mapping, modes_functional
        else:
            return mode_similarity, mapping

    def compute_correlation_curve(self, C, d_max=0.5, n_bins=21):
        distances = compute_distances(self.coordinates, self.coordinates)
        bins = np.linspace(0, d_max, n_bins, endpoint=True)
        avg_corr_per_dist = []
        std_corr_per_dist = []
        for i in range(len(bins) - 1):
            corrs_in_bin = C[(distances > bins[i]) & (distances <= bins[i + 1])]
            avg_corr_per_dist.append(np.mean(corrs_in_bin))
            std_corr_per_dist.append(np.std(corrs_in_bin))
        return avg_corr_per_dist, std_corr_per_dist

    def compute_structure_function_coupling(self, C):
        triangle = np.triu_indices(C.shape[0], 1)
        return pearsonr(C[triangle], np.abs(self.torch_to_numpy(self.dynamics.W)[triangle]))[0]

    def integrate(self, T, output=True):
        if output:
            return self.torch_to_numpy(self.dynamics.integrate(T))
        else:
            self.timeseries = self.torch_to_numpy(self.dynamics.integrate(T))

    def plot(self, N_neurons=30, spacing=1):
        if self.timeseries is None:
            self.integrate(output=False)
        plt.figure(figsize=(12, 6))
        for i in range(N_neurons):
            plt.plot(self.normalize(self.timeseries[i]) - spacing * i, color='black')
        plt.axis('off')
        plt.show()

    def imshow(self, cmap='hot', vmin=None, vmax=None):
        if self.timeseries is None:
            self.integrate(output=False)
        plt.figure(figsize=(12, 6))
        if vmin is None:
            vmin = np.min(self.timeseries)
        elif vmax is None:
            vmax = np.max(self.timeseries)
        plt.imshow(self.timeseries, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.show()

    def animate(self, cmap='hot', vmin=None, vmax=None, fps=50, s=50, alpha=0.5, lim=1, rotation_speed=0.5, elev=30):
        coords = self.coordinates
        if self.timeseries is None:
            self.integrate(output=False)
        if vmin is None:
            vmin = np.min(self.timeseries)
        elif vmax is None:
            vmax = np.max(self.timeseries)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        delta = lim / 2
        sc = ax.scatter(coords[:, 0] + delta, coords[:, 1] + delta, coords[:, 2] + delta,
                        c=self.timeseries[:, 0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=s,
                        alpha=alpha,
                        edgecolor='None')
        ax.set_xlim([0, lim])
        ax.set_ylim([0, lim])
        ax.set_zlim([0, lim])
        ax.axis('off')

        def update(frame):
            sc.set_array(self.timeseries[:, frame])  # Update colors based on activity
            # ax.set_title(f"Time Step: {frame}")
            ax.view_init(elev=elev, azim=frame * rotation_speed)  # Adjust rotation speed with the multiplier
            return sc,

        self.animation = FuncAnimation(fig, update, frames=self.timeseries.shape[1], interval=1000 / fps, blit=False)
        plt.show()

    def save_animation(self, path, fps=50):
        self.animation.save(path, writer='pillow', fps=fps)

    @staticmethod
    def normalize(array):
        normalized = np.copy(array)
        normalized -= np.min(normalized)
        if np.max(normalized) != 0:
            normalized /= np.max(normalized)
        return normalized

    @staticmethod
    def torch_to_numpy(tensor):
        try:
            return tensor.detach().cpu().numpy()
        except:  # Assumes the tensor is already a numpy array
            return tensor


class ChaoticRNN:

    def __init__(self, coordinates, params, device='cuda:0'):
        self.device = device

        self.coordinates = coordinates
        N = self.coordinates.shape[0]
        self.tau = torch.tensor(params['tau']).float().to(device)
        self.g = torch.tensor(params['g']).float().to(device)
        self.h = params['h']
        self.dale = params['dale']

        W = generate_connectivity_matrix(coordinates, h=self.h, dale=self.dale)
        self.W = torch.tensor(W).float().to(device)

        self.x_0 = torch.from_numpy(np.random.uniform(-1, 1, (N, 1))).float().to(device)
        self.r_0 = self.phi(self.x_0)
        self.X = [self.x_0]
        self.R = [self.r_0]

    def integrate(self, T):
        self.X = [self.x_0]
        self.R = [self.r_0]
        for t in range(T):  # Euler integration scheme
            r = self.R[-1]
            x = self.X[-1]
            dx = (1 / self.tau) * (-x + self.g * torch.matmul(self.W, r))
            self.X.append(x + dx)
            self.R.append(self.phi(self.X[-1]))
        return torch.concatenate(self.R, axis=1)

    def phi(self, x):
        return torch.tanh(x)


class AdjustedChaoticRNN:

    def __init__(self, coordinates, params, device='cuda:0'):
        self.device = device

        self.coordinates = coordinates
        N = self.coordinates.shape[0]
        self.tau = torch.tensor(params['tau']).float().to(device)
        self.g = torch.tensor(params['g']).float().to(device)
        self.x_s = params['x_s']
        self.dale = params['dale']
        self.h = params['h']
        self.g_I = params['g_I']

        W = generate_connectivity_matrix(self.coordinates, h=self.h, dale=self.dale)
        self.W = torch.tensor(W).float().to(device)
        self.W[self.W < 0] *= self.g_I

        self.x_0 = torch.from_numpy(np.random.uniform(-1, 1, (N, 1))).float().to(device)
        self.r_0 = self.phi(self.x_0)
        self.X = [self.x_0]
        self.R = [self.r_0]

    def integrate(self, T):
        self.X = [self.x_0]
        self.R = [self.r_0]
        for t in range(T):  # Euler integration scheme
            r = self.R[-1]
            x = self.X[-1]
            dx = (1 / self.tau) * -x + self.g * torch.matmul(self.W, r)
            self.X.append(x + dx)
            self.R.append(self.phi(self.X[-1]))
        return torch.concatenate(self.R, axis=1)

    def phi(self, x):
        return torch.where(x > 0,
                           (2 - self.x_s) * torch.tanh(x / (2 - self.x_s)),
                           self.x_s * torch.tanh(x / self.x_s))


class KuramotoSakaguchi:

    def __init__(self, coordinates, params, device='cuda:0'):
        self.device = device

        self.coordinates = coordinates
        self.N = coordinates.shape[0]
        self.alpha = params['alpha']
        self.tau = params['tau']
        self.k = params['coupling']
        self.h = params['h']
        self.std = params['std']

        W = self.generate_random_network(coordinates, h=self.h)
        self.W = torch.tensor(W).float().to(device)
        self.distances = compute_distances(coordinates, coordinates)
        self.phase_lags = torch.tensor((W > 0).astype('float') * self.distances * self.alpha).float().to(device)

        self.omega = torch.from_numpy(np.random.normal(2, self.std, (self.N, 1))).float().to(device)
        self.theta_0 = torch.from_numpy(np.random.uniform(0, 2 * np.pi, (self.N, 1))).float().to(device)
        self.theta = [self.theta_0]

    def integrate(self, T):
        self.theta = [self.theta_0]
        for _ in range(T - 1):
            theta = self.theta[-1]
            phase_differences = theta - theta.T
            dtheta = (1 / self.tau) * (self.omega + (1 / self.N) * torch.sum(self.k * self.W * torch.sin(phase_differences - self.phase_lags), axis=1)[:, None])
            self.theta.append(theta + dtheta)
        return torch.sin(torch.concatenate(self.theta, axis=1))

    @staticmethod
    def generate_random_network(coordinates, h=0.1):
        distances = compute_distances(coordinates, coordinates)
        W = (distances <= h).astype('float')
        W[np.diag_indices(W.shape[0])] = 0
        return W


class BinaryNetwork:

    def __init__(self, coordinates, params, device='cuda:0'):

        self.device = device

        self.N = coordinates.shape[0]
        self.coordinates = coordinates
        self.h = params['h']
        self.P_activation = params['P_activation']
        self.P_deactivation = params['P_deactivation']

        W = self.generate_random_network(self.coordinates, h=self.h)
        self.W = torch.tensor(W).float().to(self.device)
        self.X_0 = torch.from_numpy((np.random.uniform(0, 1, (self.N, 1)) > 0.9).astype('float')).float().to(device)
        self.X = [self.X_0]

    def integrate(self, T):
        self.X = [self.X_0]
        for _ in range(T - 1):
            X = self.X[-1]
            activation = (
                        (torch.matmul(self.W, X) * self.P_activation) > torch.rand((self.N, 1)).to(self.device)).float().to(
                self.device)
            no_deactivation = (self.P_deactivation < torch.rand((self.N, 1)).to(self.device)).float().to(self.device)
            # no_deactivation[activation == 1] = 1
            X_new = (((X + activation) * no_deactivation) > 0).float().to(self.device)
            self.X.append(X_new)
        return torch.concatenate(self.X, axis=1)

    @staticmethod
    def generate_random_network(coordinates, h=0.1):
        distances = compute_distances(coordinates, coordinates)
        W = (distances <= h).astype('float')
        W[np.diag_indices(W.shape[0])] = 0
        signs = ([-1] * int(W.shape[0] / 2)) + ([1] * int(W.shape[0] / 2))
        np.random.shuffle(signs)
        for i in range(W.shape[1]):
            W[:, i] = signs[i] * np.abs(W[:, i])
        return W


class LIFNetwork:

    def __init__(self, coordinates, params, device=None):

        self.device = device
        self.N = coordinates.shape[0]
        self.coordinates = coordinates

        # Constants
        self.g_L = params['g_L']
        self.E_L, self.E_E, self.E_I = params['E_L'], params['E_E'], params['E_I']
        self.V_thresh, self.V_reset = params['V_thresh'], params['V_reset']
        self.tau = params['tau']
        self.I_E, self.I_I = params['I_E'], params['I_I']
        self.sigma_E, self.sigma_I = params['sigma_E'], params['sigma_I']
        self.delta_t = params['delta_t']
        self.h = params['h']

        # Voltage matrices
        self.V = np.expand_dims([self.V_reset] * self.N, axis=1)

        # Spike matrices
        self.spikes = np.expand_dims([0] * self.N, axis=1)

        # Conductance matrices
        self.g_EE = np.expand_dims([0] * int(self.N / 2), axis=1)
        self.g_II = np.expand_dims([0] * int(self.N / 2), axis=1)
        self.g_EI = np.expand_dims([0] * int(self.N / 2), axis=1)
        self.g_IE = np.expand_dims([0] * int(self.N / 2), axis=1)

        # Synaptic connection matrices (network is initially unwired)
        self.W = self.generate_connectivity_matrix(self.coordinates, self.h)
        signs = ([-1] * int(self.N / 2)) + ([1] * int(self.N / 2))
        np.random.shuffle(signs)
        self.excitatory, self.inhibitory = np.array(signs) > 0, np.array(signs) < 0
        self.c_EE = self.W[self.excitatory, :][:, self.excitatory]
        self.c_II = self.W[self.inhibitory, :][:, self.inhibitory]
        self.c_EI = self.W[self.excitatory, :][:, self.inhibitory]
        self.c_IE = self.W[self.inhibitory, :][:, self.excitatory]

    @property
    def spike_times(self):
        times = []
        for i in range(self.N):
            times.append(np.where(self.spikes[i])[0] * self.delta_t)
        return times

    def set_input_currents(self, I_E, I_I, clamp=False):
        self.I_E, self.I_I = I_E, I_I
        if clamp:
            self.sigma_E, self.sigma_I = 0, 0
        else:
            self.sigma_E, self.sigma_I = np.sqrt(I_E), np.sqrt(I_I)

    def get_input_currents(self):
        I_E = np.array([self.I_E] * int(self.N / 2)) + np.random.normal(0, self.sigma_E, int(self.N / 2))
        I_I = np.array([self.I_I] * int(self.N / 2)) + np.random.normal(0, self.sigma_I, int(self.N / 2))
        return I_E, I_I

    def integrate(self, N_steps):

        # Initializing voltage matrices
        V = np.zeros((self.N, N_steps))

        # Initializing spike matrices
        spikes = np.zeros((self.N, N_steps))

        # Initializing conductance matrices
        g_EE, g_II = np.zeros((int(self.N / 2), N_steps)), np.zeros((int(self.N / 2), N_steps))
        g_EI, g_IE = np.zeros((int(self.N / 2), N_steps)), np.zeros((int(self.N / 2), N_steps))

        iE, iI = self.excitatory, self.inhibitory  # E/I indices

        # Propagating
        for i in range(N_steps - 1):
            # Updating currents
            I_E, I_I = self.get_input_currents()

            # Updating voltages
            V[iE, i + 1] = V[iE, i] + (self.delta_t * ((self.g_L * (self.E_L - V[iE, i])) + self.I_E
                                                       + (g_EE[:, i] * (self.E_E - V[iE, i]))
                                                       + (g_EI[:, i] * (self.E_I - V[iE, i]))))

            V[iI, i + 1] = V[iI, i] + (self.delta_t * ((self.g_L * (self.E_L - V[iI, i])) + self.I_I
                                                       + (g_II[:, i] * (self.E_I - V[iI, i]))
                                                       + (g_IE[:, i] * (self.E_E - V[iI, i]))))

            # Applying reset voltages and updating spikes
            V[V[:, i] >= self.V_thresh, i + 1] = self.V_reset
            spikes[V[:, i] >= self.V_thresh, i] = 1

            # Updating conductances
            g_EE[:, i + 1] = g_EE[:, i] * (1 - (self.delta_t / self.tau)) + self.c_EE @ spikes[iE, i]
            g_EI[:, i + 1] = g_EI[:, i] * (1 - (self.delta_t / self.tau)) + self.c_EI @ spikes[iI, i]
            g_II[:, i + 1] = g_II[:, i] * (1 - (self.delta_t / self.tau)) + self.c_II @ spikes[iI, i]
            g_IE[:, i + 1] = g_IE[:, i] * (1 - (self.delta_t / self.tau)) + self.c_IE @ spikes[iE, i]

        # Storing data in object attributes
        self.V = np.append(self.V, V[:, 1:], axis=1)
        self.spikes = np.append(self.spikes, spikes[:, 1:], axis=1)
        self.g_EE = np.append(self.g_EE, g_EE[:, 1:], axis=1)
        self.g_EI = np.append(self.g_EI, g_EI[:, 1:], axis=1)
        self.g_II = np.append(self.g_II, g_II[:, 1:], axis=1)
        self.g_IE = np.append(self.g_IE, g_IE[:, 1:], axis=1)

        return self.spikes

    @staticmethod
    def generate_connectivity_matrix(coordinates, h=0.1):
        distances = compute_distances(coordinates, coordinates)
        N = coordinates.shape[0]
        W = (distances <= h).astype('float')
        W[np.diag_indices(N)] = 0
        N_edges = np.sum(W != 0)
        return W


class RNN(nn.Module):

    def __init__(self, N, coordinates, tau=5, g=1.5, alpha=5, beta=0.5, h=0.1, device='cuda:0', dale=False, W_0=None):

        self.device = device

        self._N = N
        self.tau = torch.tensor(tau).float().to(device)
        self.g = torch.tensor(g).float().to(device)
        self.alpha = torch.tensor(alpha).float().to(device)
        self.beta = torch.tensor(beta).float().to(device)

        self.coordinates = coordinates

        if dale:
            self.signs = np.sign(np.random.uniform(-1, 1, (N,)))
            W = self.signs * np.abs(self.generate_random_network(N, coordinates, h=h, W_0=W_0))
        else:
            W = self.generate_random_network(N, coordinates, h=h, W_0=W_0)
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
        #return 1 / (1 + torch.exp(-self.alpha * (x - self.beta))) # Logistic function

    def reset_initial_conditions(self):
        self.x_0 = torch.from_numpy(np.random.uniform(-1, 1, (self._N, 1))).float().to(self.device)
        self.r_0 = self.phi(self.x_0)

    @staticmethod
    def generate_random_network(N, coordinates, h=0.1, W_0=None):
        distances = compute_distances(coordinates, coordinates)
        if W_0 is not None:
            W = (W_0 > 0).astype('float')
        else:
            W = (distances <= h).astype('float')
        W[np.diag_indices(N)] = 0
        N_edges = np.sum(W > 0)
        rho = N_edges / (N * (N - 1))
        W[W != 0] = np.random.normal(0, np.sqrt(1 / (N * rho)), N_edges)
        return W


# ----------------------------------------------------------------------------------------------------------------------
# Various utilities functions
# ----------------------------------------------------------------------------------------------------------------------


def normalize(array):
    normalized = np.copy(array)
    normalized = normalized - np.min(normalized)
    normalized = normalized / np.max(normalized)
    return normalized


def generate_connectivity_matrix(coordinates, h=0.1, dale=False):
    distances = compute_distances(coordinates, coordinates)
    N = coordinates.shape[0]
    W = (distances <= h).astype('float')
    W[np.diag_indices(N)] = 0
    N_edges = np.sum(W != 0)
    rho = N_edges / (N * (N - 1))
    W[W != 0] = np.random.normal(0, np.sqrt(1 / (N * rho)), N_edges)
    if dale:
        signs = ([-1] * int(N / 2)) + ([1] * int(N / 2))
        np.random.shuffle(signs)
        W = np.array(signs) * np.abs(W)
    if dale:
        for i in range(W.shape[0]): # Balancing E/I input strengths
            E = np.sum(W[i][W[i] > 0])
            I = np.abs(np.sum(W[i][W[i] < 0]))
            ratio = E / I
            W[i][W[i] > 0] = W[i][W[i] > 0] / ratio
    return W


def progress_bar(iterable, verbose=True):
    if verbose:
        wrapped_iterable = tqdm(iterable)
    else:
        wrapped_iterable = iterable
    for item in wrapped_iterable:
        yield item


def gcampify(data, tau, dt=1.0):
    n_signals, n_timepoints = data.shape
    kernel_length = int(10 * tau / dt)  # Kernel length: 10 times the decay constant
    time = np.arange(0, kernel_length * dt, dt)
    kernel = np.exp(-time / tau)
    kernel /= kernel.sum()  # Normalize the kernel
    gcamp_data = np.array([
        np.convolve(row, kernel, mode='full')[:n_timepoints] for row in data
    ])
    return gcamp_data


def compute_distances(X, Y):
    """
    Computes the pairwise distances between the rows of two numpy arrays.

    Parameters:
    X (np.ndarray): A numpy array of shape (n_samples_X, n_features).
    Y (np.ndarray): A numpy array of shape (n_samples_Y, n_features).

    Returns:
    np.ndarray: A numpy array of shape (n_samples_X, n_samples_Y) where
                each element (i, j) is the distance between X[i] and Y[j].
    """
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    return distances
    
    
def compute_mode_similarity_matrix(modes1, modes2, return_mapping=False):
    N_modes = modes1.shape[0]
    corrs = np.zeros((N_modes, N_modes))
    for i in range(N_modes):
        for j in range(N_modes):
            corrs[i, j] = pearsonr(modes1[i], modes2[j])[0]
    
    cost_matrix = 1 - np.abs(corrs)
    _, mapping = linear_sum_assignment(cost_matrix)
    
    corrs = np.zeros((N_modes, N_modes))
    for i in range(N_modes):
        for j in range(N_modes):
            corrs[i, j] = pearsonr(modes1[i], modes2[mapping[j]])[0]
    if return_mapping:
        return corrs, mapping
    else:
        return corrs
        

def gaussian_kernel(distance, sigma):
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def spatial_smoothing(time_series, node_coordinates, sigma=0.1):
    if sigma != 0:
        distances = cdist(node_coordinates, node_coordinates, 'euclidean')
        weights = gaussian_kernel(distances, sigma)
        weights_normalized = weights / weights.sum(axis=1, keepdims=True)
        smoothed_time_series = np.dot(weights_normalized, time_series)
        return smoothed_time_series
    else:
        return time_series
    

def compute_pairwise_distances(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances


def compute_variogram(coordinates, values, bins=np.linspace(0, 1, 30), subsample=1000, iters=10):
    variograms = []
    for _ in range(iters):
        random_ids = np.arange(coordinates.shape[0])
        np.random.shuffle(random_ids)
        random_ids = np.sort(random_ids[:subsample])
        d = compute_pairwise_distances(coordinates[random_ids])
        sub_values = values[random_ids]
        variances = (sub_values[np.newaxis, :] - sub_values[:, np.newaxis]) ** 2
        variogram = np.zeros((len(bins) - 1, ))
        for i in range(len(variogram)):
            variogram[i] = np.mean(variances[(d >= bins[i]) & (d < bins[i + 1])])
        variograms.append(variogram)
    return np.mean(np.stack(variograms, axis=0), axis=0)


def get_wavelength(variogram, bins):
    result = find_peaks(variogram)
    if any(result[0]):
        wavelength = 2 * (bins[find_peaks(variogram)[0][0]] + ((bins[1] - bins[0]) / 2))
        return wavelength
    else:
        return 2 * bins[-1]
        
        
def quantify_mapping_quality(mapping_matrix, cutoff=None, offdiag=False):
    if cutoff is None:
        cutoff = mapping_matrix.shape[0]
    avg_diag = np.mean(np.abs(np.diag(mapping_matrix))[:cutoff])
    if offdiag:
        triangle = np.triu_indices(cutoff, 1)
        offdiag1 = mapping_matrix[triangle]
        offdiag2 = mapping_matrix.T[triangle]
        offdiag = np.concatenate([offdiag1, offdiag2])
        highest_offdiag = offdiag[np.argsort(offdiag)][-cutoff:]
        avg_offdiag = np.mean(np.abs(highest_offdiag))
        return avg_diag / avg_offdiag
    else:
        return avg_diag


def rotate_points_around_z(points, theta):
    """
    Rotates a set of 3D points around the Z-axis by an angle theta.

    Parameters:
    points (np.ndarray): A numpy array of shape (N, 3) representing the 3D points.
    theta (float): The angle of rotation in radians.

    Returns:
    np.ndarray: A numpy array of the rotated points, same shape as input.
    """
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points


def find_nearest_neighbors(coord_set1, coord_set2):
    """
    Finds the nearest neighbors in coord_set2 for each point in coord_set1.

    Parameters:
    coord_set1 (np.ndarray): A numpy array of shape (N, 3) representing the first set of 3D coordinates.
    coord_set2 (np.ndarray): A numpy array of shape (M, 3) representing the second set of 3D coordinates.

    Returns:
    np.ndarray: An array of indices representing the nearest neighbor in coord_set2 for each point in coord_set1.
    np.ndarray: An array of distances to the nearest neighbors for each point in coord_set1.
    """
    # Create a KDTree for the second set of coordinates
    tree = cKDTree(coord_set2)

    # Query the KDTree to find the nearest neighbor for each point in coord_set1
    distances, indices = tree.query(coord_set1)

    return indices, distances


def find_optimal_rotation(ellipse, functional_gradients, N_modes=5):
    coords = np.copy(ellipse.vertices[ellipse.random_ids])
    vertices = np.copy(ellipse.vertices)
    eigenmodes = np.copy(ellipse.eigenmodes)

    # Temporarily centering coordinates on the z-axis (x=0, y=0)
    coords[:, 0] = coords[:, 0] - np.mean(vertices[:, 0])
    coords[:, 1] = coords[:, 1] - np.mean(vertices[:, 1])
    vertices[:, 0] = vertices[:, 0] - np.mean(vertices[:, 0])
    vertices[:, 1] = vertices[:, 1] - np.mean(vertices[:, 1])

    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    corr_values = []
    for theta in angles:
        # Rotating subsampled ellipse coordinates
        coords_r = rotate_points_around_z(coords, theta)
        # Finding nearest neighbors in ellipse vertices
        nn_ids, _ = find_nearest_neighbors(coords_r, vertices)

        # Comparing rotated gradients with ellipse eigenmodes
        modes1 = functional_gradients[1:N_modes + 1]  # Skipping first mode which is rotation invariant
        modes2 = eigenmodes[2:N_modes + 2, nn_ids]
        corr_matrix = compute_mode_similarity_matrix(modes1, modes2)

        corr_values.append(np.mean(np.diag(np.abs(corr_matrix))))

    theta_opt = angles[np.argmax(corr_values)]
    coords_r = rotate_points_around_z(coords, theta_opt)
    nn_ids, _ = find_nearest_neighbors(coords_r, vertices)
    modes1 = functional_gradients
    modes2 = eigenmodes[1:modes1.shape[0] + 1, nn_ids]
    corr_matrix = compute_mode_similarity_matrix(modes1, modes2)
    coords_r[:, 0] = coords[:, 0] + np.mean(ellipse.vertices[:, 0])
    coords_r[:, 1] = coords[:, 1] + np.mean(ellipse.vertices[:, 1])

    return coords_r, corr_matrix


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------


def groups_ANOVA_Tukey(data, significance_ANOVA=0.05):
    """
    Data: list of 1D numpy arrays.
    """
    f_value, p_value = f_oneway(*data)
    print(f'ANOVA results: F = {f_value}, p = {p_value}')
    print('')
    if p_value < significance_ANOVA:
        data_flat = np.concatenate(data)
        groups = np.array([])
        for i in range(len(data)):
            groups = np.append(groups, [f'Group {i+1}'] * len(data[i]))
        tukey = mc.MultiComparison(data_flat, groups)
        result = tukey.tukeyhsd()
        print(result.summary())


def linear_regression_with_confidence_interval(x, y, n_iter=10000, fraction=0.5):
    a, b = np.polyfit(x, y, deg=1)

    a_distribution, b_distribution = [], []
    for _ in range(n_iter):
        selected_ids = np.random.choice(np.arange(len(x)), int(len(x) * fraction), replace=False)
        a_, b_ = np.polyfit(x[selected_ids], y[selected_ids], deg=1)
        a_distribution.append(a_)
        b_distribution.append(b_)

    CI_a = [np.percentile(a_distribution, 2.5), np.percentile(a_distribution, 97.5)]
    CI_b = [np.percentile(b_distribution, 2.5), np.percentile(b_distribution, 97.5)]

    return (a, b), CI_a, CI_b


def get_linear_bounds(x, CI_a, CI_b):
    regressions = np.stack([CI_a[0] * x + CI_b[0],
                            CI_a[1] * x + CI_b[0],
                            CI_a[0] * x + CI_b[1],
                            CI_a[1] * x + CI_b[1]], axis=0)
    upper_bound = np.max(regressions, axis=0)
    lower_bound = np.min(regressions, axis=0)
    return lower_bound, upper_bound


def compute_r_squared(y_data, y_model):
    ss_res = np.sum((y_data - y_model) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def piecewise_linear(x, x_break, slope, intercept):
    y1 = slope * (x - x_break) + intercept
    y2 = intercept
    return np.where(x < x_break, y1, y2)


# ----------------------------------------------------------------------------------------------------------------------
# Functions associated with the double edge swapping algorithm
# ----------------------------------------------------------------------------------------------------------------------


def save_graph_to_file(adjacency_matrix: np.ndarray, filename: str, distance_matrix: np.ndarray = None) -> None:
    """
    Saves the graph represented by the adjacency matrix to a file in CSV format.

    Args:
    - adjacency_matrix (np.ndarray): The adjacency matrix representing the graph.
    - filename (str): The name of the file to save the graph to.
    - distance_matrix (np.ndarray, optional): The distance matrix representing the distances between nodes. Defaults to None.

    Returns:
        None
    """
    # Checks if the graph is undirected
    if np.allclose(adjacency_matrix, adjacency_matrix.T):
        undirected = True
    else:
        undirected = False

    # Checks if the graph is unweighted
    if np.array_equal(adjacency_matrix, adjacency_matrix.astype(bool)):
        weighted = False
    else:
        weighted = True

    # Extracts the edgelist
    if undirected:
        edgelist = np.argwhere(np.triu(adjacency_matrix))
    else:
        edgelist = np.argwhere(adjacency_matrix)

    header = ["source", "target"]
    df = {}
    df["source"] = edgelist[:, 0]
    df["target"] = edgelist[:, 1]

    # Appends the weights, if any
    if weighted:
        header.append("weight")
        edgelist = np.c_[edgelist, adjacency_matrix[edgelist[:, 0], edgelist[:, 1]]]

    # Appends the distances, if any
    if distance_matrix is not None:
        header.append("distance")
        edgelist = np.c_[edgelist, distance_matrix[edgelist[:, 0], edgelist[:, 1]]]

    # Saves the edgelist as a csv text file
    df = pd.DataFrame(edgelist, columns=header)
    df.to_csv(filename, index=False)


def within_distance_interval(target1: int, source1: int, target2: int, source2: int, distance_matrix: np.ndarray,
                             min_distance: float = None, max_distance: float = None) -> bool:
    """
    Checks if the distances between the given pairs of target and source vertices are within the specified distance interval.

    Args:
    - target1 (int): The index of the first target vertex.
    - source1 (int): The index of the first source vertex.
    - target2 (int): The index of the second target vertex.
    - source2 (int): The index of the second source vertex.
    - distance_matrix (numpy.ndarray): The matrix containing the distances between vertices.
    - min_distance (float, optional): The minimum distance allowed.
    - max_distance (float, optional): The maximum distance allowed.

    Returns:
    - bool: True if the distances between the given pairs of nodes are within the specified distance interval, False otherwise.
    """
    if min_distance != None:
        if distance_matrix[target1, source1] < min_distance:
            return False
        if distance_matrix[target2, source2] < min_distance:
            return False
        if distance_matrix[target1, source2] < min_distance:
            return False
        if distance_matrix[target2, source1] < min_distance:
            return False
    if max_distance != None:
        if distance_matrix[target1, source1] > max_distance:
            return False
        if distance_matrix[target2, source2] > max_distance:
            return False
        if distance_matrix[target1, source2] > max_distance:
            return False
        if distance_matrix[target2, source1] > max_distance:
            return False
    return True


def double_edge_swap(adjacency_matrix: np.ndarray, weights: str = None, number_of_iterations: int = None,
                     distance_matrix: np.ndarray = np.zeros((1, 1)), min_distance: float = None,
                     max_distance: float = None) -> np.ndarray:
    """
    Performs the double-edge swap algorithm on a given adjacency matrix.

    Args:
    - adjacency_matrix (numpy.ndarray): The adjacency matrix representing the graph.
    - weights (str, optional): Indicates how to handle the weights of the edges. If 'random', the two original weights are assigned randomly to the two new edges (default for undirected graphs). For directed graphs, 'preserve-out-strength' will assign weights to preserve the out-strength of vertices (default for directed graphs), and 'preserve-in-strength' will preserve the in-strength of vertices.
    - number_of_iterations (int, optional): The number of iterations to perform. If not provided, it is set to 10 times the number of edges in the graph.
    - distance_matrix (numpy.ndarray, optional): The distance matrix representing the distances between vertices. If provided, the swap will only be performed if the old and new edges are within the prescribed distance interval.
    - min_distance (float, optional): The minimum distance allowed for the swap. Only applicable if distance_matrix is provided.
    - max_distance (float, optional): The maximum distance allowed for the swap. Only applicable if distance_matrix is provided.

    Returns:
    - numpy.ndarray: The updated adjacency matrix after performing the double-edge swap algorithm.

    References:
    - B. K. Fosdick, D. B. Larremore, J. Nishimura, and J. Ugander, Configuring Random Graph Models with Fixed Degree Sequences, SIAM Review, vol. 60, pp. 315–355, 2018, doi: 10.1137/16M1087175.
    """
    # Makes a copy of the adjacency matrix to avoid modifying the original
    adjacency_matrix = copy.deepcopy(adjacency_matrix)

    # Checks is the graph is undirected
    if np.allclose(adjacency_matrix, adjacency_matrix.T):
        undirected = True
    else:
        undirected = False

    # Checks is the graph is unweighted
    if np.array_equal(adjacency_matrix, adjacency_matrix.astype(bool)):
        weighted = False
    else:
        weighted = True

    # Creates a list of edges (should be faster than working with the adjacency matrix)
    if undirected:
        edgelist = np.argwhere(np.triu(adjacency_matrix))
        if weighted:
            if weights == None:
                weights = 'random'
    else:
        edgelist = np.argwhere(adjacency_matrix)
        if weighted:
            if weights == None:
                weights = 'preserve-out-strength'
    number_of_edges = edgelist.shape[0]

    # Sets the number of iterations if none is provided
    if number_of_iterations == None:
        number_of_iterations = 10 * number_of_edges

    # Performs the double-edge swap algorithm
    number_of_iterations_completed = 0
    while number_of_iterations_completed < number_of_iterations:

        # Chooses two distinct edges at random and identifies the vertices
        e1, e2 = np.random.choice(number_of_edges, 2, replace=False)
        target1, source1 = edgelist[e1, 0], edgelist[e1, 1]
        target2, source2 = edgelist[e2, 0], edgelist[e2, 1]
        weight1 = adjacency_matrix[target1, source1]
        weight2 = adjacency_matrix[target2, source2]

        # Randomly choose one of the two possible swaps
        if undirected:
            if np.random.rand() < 0.5:
                source2, target2 = target2, source2

        # Check that the old and new edges are within a prescribed distance interval
        if not np.allclose(distance_matrix, np.zeros((1, 1))):
            if not within_distance_interval(target1, source1, target2, source2, distance_matrix, min_distance,
                                            max_distance):
                continue

        # Registers that an iteration has been completed
        number_of_iterations_completed += 1

        # Check if the swap would leave the graph space
        if target1 == source2:  # Would create a self-loop
            continue
        if target2 == source1:  # Would create a self-loop
            continue
        if not np.isclose(adjacency_matrix[target1, source2], 0):  # Would create a parallel edge
            continue
        if not np.isclose(adjacency_matrix[target2, source1], 0):  # Would create a parallel edge
            continue

        # Performs the swap by updating the adjacency matrix and the edgelist
        adjacency_matrix[target1, source1] = 0
        adjacency_matrix[target2, source2] = 0
        if undirected:
            adjacency_matrix[source1, target1] = 0
            adjacency_matrix[source2, target2] = 0

        if weighted:
            if weights == 'random':
                if np.random.rand() < 0.5:
                    weight1, weight2 = weight2, weight1
            elif weights == 'preserve-out-strength':
                weight1, weight2 = weight2, weight1
            elif weights == 'preserve-in-strength':
                pass
            else:
                raise ValueError(
                    "The 'weights' parameter must be either 'random', 'preserve-out-strength', or 'preserve-in-strength'.")

        adjacency_matrix[target1, source2] = weight1
        adjacency_matrix[target2, source1] = weight2
        if undirected:
            adjacency_matrix[source2, target1] = weight1
            adjacency_matrix[source1, target2] = weight2

        edgelist[e1, 0], edgelist[e1, 1] = target1, source2
        edgelist[e2, 0], edgelist[e2, 1] = target2, source1

    return adjacency_matrix


@njit
def double_edge_swap_fraction(W, D, fraction=1.0, min_distance=None, max_distance=None, replace=False):
    """Even if W matrix is symmetrical, treats the network as undirected."""

    adjacency_matrix = np.copy(W).astype('float')

    if min_distance is None:
        min_distance = 0
    if max_distance is None:
        max_distance = D.max()

    edgelist = np.argwhere(W)
    n_edges = edgelist.shape[0]
    n_swapped_edges = int(fraction * n_edges)
    n_swaps = int(n_swapped_edges / 2)

    swapped_edges = np.zeros((n_edges,)).astype('bool')
    while n_swaps > 0:

        # Chooses two distinct edges at random and identifies the vertices
        if replace:
            e1, e2 = np.random.choice(n_edges, 2, replace=False)
        else:
            unswapped_edges = np.where(swapped_edges == False)[0]
            e1, e2 = np.random.choice(unswapped_edges, 2, replace=False)
        target1, source1 = edgelist[e1, 0], edgelist[e1, 1]
        target2, source2 = edgelist[e2, 0], edgelist[e2, 1]
        weight1 = adjacency_matrix[target1, source1]
        weight2 = adjacency_matrix[target2, source2]

        # Checking that the new edges don't interfere with existing ones
        if target1 == source2:  # Would create a self-loop
            continue
        if target2 == source1:  # Would create a self-loop
            continue
        if not adjacency_matrix[target1, source2] == 0:  # Would create a parallel edge
            continue
        if not adjacency_matrix[target2, source1] == 0:  # Would create a parallel edge
            continue
        # Checking that the new edges respect distance constraints
        d1, d2 = D[target1, source2], D[target2, source1]
        if (d1 < min_distance) or (d2 < min_distance):
            continue
        if (d1 > max_distance) or (d2 > max_distance):
            continue

        if (d1 > min_distance) and (d2 > min_distance):  # A bit redundant with previous verifications but whatever
            adjacency_matrix[target1, source1] = 0
            adjacency_matrix[target2, source2] = 0
            adjacency_matrix[target1, source2] = weight1
            adjacency_matrix[target2, source1] = weight2
            edgelist[e1, 0], edgelist[e1, 1] = target1, source2
            edgelist[e2, 0], edgelist[e2, 1] = target2, source1

            swapped_edges[e1], swapped_edges[e2] = True, True
            n_swaps -= 1

    return adjacency_matrix


def ave_connectivity(A):  # Once again, weird if negative weights
    return np.mean([np.sum(row) for row in A])


def ave_connection_distance(A, distances):
    weighted_distances = np.multiply(A, distances)
    return np.mean([np.sum(row) for row in weighted_distances])


def single_weight_swap(A, distances, h=0.3, iterations=5):
    N = len(A)
    long_edge = random.sample(range(N), 2)
    while A[long_edge[0]][long_edge[1]] == 0 or distances[long_edge[0]][long_edge[1]] < h:
        long_edge = random.sample(range(N), 2)

    i = 0
    while A[long_edge[0]][long_edge[1]] > 0 and i < iterations:
        new_edge = random.sample(range(N), 2)
        while distances[new_edge[0]][new_edge[1]] > distances[long_edge[0]][long_edge[1]] or long_edge == (
                new_edge[0], new_edge[1]):
            new_edge = random.sample(range(N), 2)

        distance_ratio = distances[new_edge[0]][new_edge[1]] / distances[long_edge[0]][long_edge[1]]

        if A[new_edge[0]][new_edge[1]] > 0:
            A_long_diff = min(A[long_edge[0]][long_edge[1]], A[new_edge[0]][new_edge[1]] * distance_ratio)
            A[new_edge[0]][new_edge[1]] += min(A[long_edge[0]][long_edge[1]] / distance_ratio,
                                               A[new_edge[0]][new_edge[1]])
        elif A[new_edge[0]][new_edge[1]] == 0:
            A_long_diff = min(A[long_edge[0]][long_edge[1]], distance_ratio)
            A[new_edge[0]][new_edge[1]] += min(A[long_edge[0]][long_edge[1]] / distance_ratio, 1)
        A[long_edge[0]][long_edge[1]] -= A_long_diff

        i += 1
    return np.array(A)


def weight_swapping_procedure(W, distances, h=0.1, n_iter=10000):
    W_swapped = np.copy(W)
    for i in range(n_iter):
        W_swapped = single_weight_swap(np.abs(W_swapped), distances, h=h)
    N_edges = np.sum(W_swapped > 0)
    signs = [-1] * int(N_edges / 2) + [1] * (N_edges - int(N_edges / 2))
    np.random.shuffle(signs)
    W_swapped[W_swapped != 0] *= signs
    return W_swapped