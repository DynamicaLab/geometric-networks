from main import *

# Torch stuff
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

# Loading data
vertices = np.load('tectum_vertices_right.npy') * 40
vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=1)
eigenmodes = np.load('tectum_eigenmodes_right.npy')[1:]
tectum = Geometry(vertices, eigenmodes.T)

# Dynamical parameters
params = {'N_neurons': 2850,
          'h': 0.1,
          'gamma': 0.05,
          'g': 3,
          'tau': 3,
          'dale': False,
          'rule': 'EDR'
          }

# gamma sweep parameters
gamma_values = np.linspace(0.01, 0.25, 100, endpoint=True)[50:75]
N_runs = 10
N_average = 50
N_modes = 50

mode_correlations_per_gamma = []

for gamma in tqdm(gamma_values):

    params['gamma'] = gamma
    matrices_per_run = []

    for _ in range(N_runs):

        simulator = Simulator(tectum,
                              ChaoticRNN,
                              params)

        # Running multiple simulations to generate average correlation matrix
        W_0 = numpify(simulator.dynamics.W)
        C = simulator.compute_average_correlations(n_iters=N_average,
                                                   T=2500,
                                                   W_0=W_0,
                                                   verbose=False)

        # Computing stuff from average correlation matrix C
        mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=N_modes)
        
        matrices_per_run.append(mode_similarity)
        
    mode_correlations_per_gamma.append(matrices_per_run)
    
np.save('mode_correlations_per_gamma_2500_part3.npy', mode_correlations_per_gamma)
