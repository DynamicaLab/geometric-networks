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
          'gamma': 20,
          'g': 3,
          'tau': 3,
          'dale': False,
          'rule': 'EDR'
          }

# gamma sweep parameters
gamma_values = np.logspace(0, np.log(60), 100, base=np.e, endpoint=True)[:25]
N_runs = 10
N_average = 50
N_modes = 50

mode_correlations_per_gamma = []

for gamma in tqdm(gamma_values):

    params['gamma'] = gamma
    matrices_per_run = []

    for _ in range(N_runs):

        simulator = Simulator(ellipse,
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
    
np.save('mode_correlations_per_gamma_2500_part1.npy', mode_correlations_per_gamma)
