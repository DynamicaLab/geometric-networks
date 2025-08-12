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

# Dynamical parameters
params = {'N_neurons': 1500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False
          }

# h and sigma sweep parameters
square_size = 20
sigma_values = np.linspace(0, 0.1, square_size, endpoint=True)
h_values = np.linspace(0.1, 0.6, square_size, endpoint=True)
N_runs = 10
N_average = 100
N_modes = 50
T = 500

# Simulating networks with various h and smoothing sigma levels
rows = []
for i, h in tqdm(enumerate(h_values)):
    row = []
    for j, sigma in tqdm(enumerate(sigma_values)):

        mode_similarity_matrices_ = []

        for _ in range(N_runs):
            simulator = Simulator(ellipse,
                                  ChaoticRNN,
                                  params)

            C = simulator.compute_average_correlations(n_iters=N_average,
                                                       T=T,
                                                       smoothing=True,
                                                       sigma=sigma)

            mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=N_modes)
            d = compute_distances(simulator.coordinates, simulator.coordinates)

            mode_similarity_matrices_.append(mode_similarity)
        row.append(mode_similarity_matrices_)

    rows.append(row)

np.save('mode_correlations_h_and_sigma.npy', rows)
