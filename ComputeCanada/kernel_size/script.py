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
          'dale': False
          }

# h sweep parameters
h_values = np.linspace(0.025, 0.6, 100, endpoint=True)
N_runs = 10
N_average = 100
N_modes = 50

# Simulating networks for varying N. Each "run" corresponds to a new set of 3D coordinates
# at which 100 simulations are averaged across initial conditions and connection weights.
mode_correlations_per_h = []

for h in tqdm(h_values):

    params['h'] = h
    scores_per_run = []
    matrices_per_run = []


    for _ in range(N_runs):

        simulator = Simulator(ellipse,
                              ChaoticRNN,
                              params)

        # Running multiple simulations to generate average correlation matrix
        C = simulator.compute_average_correlations(n_iters=N_average,
                                                   T=250,
                                                   verbose=False)

        # Computing stuff from average correlation matrix C
        mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=N_modes)
        
        matrices_per_run.append(mode_similarity)
        
    mode_correlations_per_h.append(matrices_per_run)
    
np.save('mode_correlations_per_h_2500_2.npy', mode_correlations_per_h)

