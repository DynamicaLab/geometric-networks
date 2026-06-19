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
params = {'N_neurons': 2500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False,
          'rule': 'EDR',
          'gamma': 25
          }

# h and sigma sweep parameters
square_size = 20
sigma_values = np.linspace(0, 0.1, square_size, endpoint=True)[4:8]
gamma_values = np.linspace(1, 27.5, square_size, endpoint=True)
N_runs = 10
N_average = 50
N_modes = 50
T = 2500

# Simulating networks with various h and smoothing sigma levels
rows = []
for i, gamma in enumerate(gamma_values):
    row = []
    params['gamma'] = gamma
    for j, sigma in tqdm(enumerate(sigma_values)):

        mode_similarity_matrices_ = []

        for _ in range(N_runs):
            simulator = Simulator(ellipse,
                                  ChaoticRNN,
                                  params)
            W_0 = numpify(simulator.dynamics.W)
            C = simulator.compute_average_correlations(n_iters=N_average,
                                                       T=T,
                                                       smoothing=True,
                                                       sigma=sigma,
                                                       W_0=W_0,
                                                       verbose=False)

            mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=N_modes)

            mode_similarity_matrices_.append(mode_similarity)
        row.append(mode_similarity_matrices_)

    rows.append(row)

np.save('mode_correlations_h_and_sigma_part2.npy', rows)
