from main import *

# Torch stuff
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

# Loading data
vertices = np.load('vertices_ellipse.npy').astype('float')
eigenmodes = np.load('eigenmodes_ellipse.npy')

ellipse = Geometry(vertices, eigenmodes)

params = {'N_neurons': 1500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False
          }

# Simulation parameters
N_modes = 50
N_runs = 10
N_average = 100

average_d_per_rho_swaps = []
mode_correlations_per_rho_swaps = []

fractions = np.linspace(0, 0.99, 36, endpoint=True)
for rho in tqdm(fractions):

    avg_per_run = []
    matrices_per_run = []
    
    for _ in range(N_runs):
        simulator = Simulator(ellipse,
                              ChaoticRNN,
                              params)
        W = simulator.dynamics.W.detach().cpu().numpy()
        D = compute_distances(simulator.coordinates, simulator.coordinates)
        W_shuffled = double_edge_swap_fraction(W, D, fraction=rho, min_distance=0.1, max_distance=1.0,
                                               replace=False)
        avg_distance = np.mean(D[W_shuffled != 0])
        avg_per_run.append(avg_distance)
        C = simulator.compute_average_correlations(n_iters=100, T=250, W_0=W_shuffled, verbose=False)
        mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=50)
        matrices_per_run.append(mode_similarity)

    average_d_per_rho_swaps.append(avg_per_run)
    mode_correlations_per_rho_swaps.append(matrices_per_run)

    np.save('avg_d_per_rho_swaps.npy', average_d_per_rho_swaps)
    np.save('mode_correlations_per_rho_swaps.npy', mode_correlations_per_rho_swaps)
