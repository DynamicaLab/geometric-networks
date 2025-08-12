from main import *

# Torch stuff
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

# Part 1: Ellipsoid ----------------------------------------------------------------------------------------------------

vertices = np.load('vertices_ellipse2.npy').astype('float')
eigenmodes = np.load('eigenmodes_ellipse2.npy')

ellipse = Geometry(vertices, eigenmodes)

params = {'N_neurons': 2500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False
          }

mode_similarity_ellipsoid = []
for _ in range(10):
    simulator = Simulator(ellipse,
                          ChaoticRNN,
                          params)
    C = simulator.compute_average_correlations(n_iters=100, T=500, verbose=False)
    mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=50)
    mode_similarity_ellipsoid.append(mode_similarity)

# Part 2: Heart --------------------------------------------------------------------------------------------------------

vertices = np.load('vertices_heart.npy').astype('float')
eigenmodes = np.load('eigenmodes_heart.npy')

heart = Geometry(vertices, eigenmodes)

params = {'N_neurons': 2500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False
          }

mode_similarity_heart = []
for _ in range(10):
    simulator = Simulator(heart,
                          ChaoticRNN,
                          params)
    C = simulator.compute_average_correlations(n_iters=100, T=500, verbose=False)
    mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=50)
    mode_similarity_heart.append(mode_similarity)

# Part 3: Torus --------------------------------------------------------------------------------------------------------

vertices = np.load('vertices_torus.npy').astype('float')
eigenmodes = np.load('eigenmodes_torus.npy')

torus = Geometry(vertices, eigenmodes)

params = {'N_neurons': 2500,
          'h': 0.1,
          'g': 3,
          'tau': 3,
          'dale': False
          }

mode_similarity_torus = []
for _ in range(10):
    simulator = Simulator(torus,
                          ChaoticRNN,
                          params)
    C = simulator.compute_average_correlations(n_iters=100, T=500, verbose=False)
    mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=50)
    mode_similarity_torus.append(mode_similarity)

# Part 4: Cow ----------------------------------------------------------------------------------------------------------

vertices = np.load('vertices_cow.npy').astype('float')
eigenmodes = np.load('eigenmodes_cow.npy')

cow = Geometry(vertices, eigenmodes)

params = {'N_neurons': 10000,
          'h': 0.05,
          'g': 5,
          'tau': 3,
          'dale': False
          }

mode_similarity_cow = []
for _ in range(10):
    simulator = Simulator(cow,
                          ChaoticRNN,
                          params)
    C = simulator.compute_average_correlations(n_iters=100, T=500, verbose=False)
    mode_similarity, _ = simulator.compute_geometric_mapping(C, N_modes=50)
    mode_similarity_cow.append(mode_similarity)

np.save('mode_similarity_ellipsoid.npy', mode_similarity_ellipsoid)
np.save('mode_similarity_heart.npy', mode_similarity_heart)
np.save('mode_similarity_torus.npy', mode_similarity_torus)
np.save('mode_similarity_cow.npy', mode_similarity_cow)
