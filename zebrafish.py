import os
import sys
import h5py
import nrrd
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from tifffile import imwrite, imread
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from matplotlib.collections import LineCollection
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.sparse import load_npz, save_npz, csc_matrix, hstack


# ----------------------------------------------------------------------------------------------------------------------
# Functions associated with locating / loading experimental data
# ----------------------------------------------------------------------------------------------------------------------


def get_datasets(top_directory, keywords=[], exclude=[], keywords_top=[]):
    folders = identify_folders(top_directory, keywords_top)
    datasets = []
    for folder in folders:
        datasets += identify_folders(folder, keywords=keywords, exclude=exclude)
    return datasets


def identify_files(path, keywords=None, exclude=None):
    items = os.listdir(path)
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    files = []
    for item in items:
        if all(keyword in item for keyword in keywords):
            if any(excluded in item for excluded in exclude):
                pass
            else:
                files.append(item)
    files.sort()
    return files


def identify_folders(path, keywords=None, exclude=None):
    initial_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    if keywords is None:
        keywords = []
    if exclude is None:
        exclude = []
    folders = []
    for folder in initial_folders:
        if all(keyword in folder for keyword in keywords):
            if any(excluded in folder for excluded in exclude):
                pass
            else:
                folders.append(folder)
    for i in range(len(folders)):
        folders[i] += '/'
    folders.sort()
    return folders


def load_data(directory):
    """
    Finds and loads a .hdf5 within a folder, assuming there is only one such file within the folder.

    Arguments:
        directory (str): Full path to a direcctory containing a .hdf5 file.

    Returns:
         dict: Dictionary containing multiple numpy arrays.
    """
    files = identify_files(directory, ['data', '.hdf5'])
    if any(files):
        if len(files) > 1:
            raise Warning('Multiple data files in directory. Loading the first one.')
        return load_hdf5(directory + files[0])
    else:
        raise FileNotFoundError('No .hdf5 data file identified in directory.')


def load_hdf5(path):
    """
    Loads a .hdf5 file into a dictionary. File must be shallow (i.e. no deep hierarchy).

    Arguments:
        path (str): Full absolute path of the .hdf5 file.

    Returns:
         data (dict): Dictionary containing multiple numpy arrays.
    """
    data = {}
    file = h5py.File(path, 'r')
    for dataset in file.keys():
        data[dataset] = np.array(file[dataset])
    file.close()
    return data


def save_hdf5(path, dictionary):
    datasets = list(dictionary.keys())
    file = h5py.File(path, 'w')
    for dataset in datasets:
        file.create_dataset(dataset, data=dictionary[dataset])
    file.close()


def load_stack(path, return_header=True):
    """
    Loads a 3D microscopy (or any other kind of)image stack. Works only for .tif and .nrrd files.
     In the second case, a file header is returned.

    Arguments:
        path (str): Full absolute path of the .tif or .nrrd file.

    Returns:
         array (np.ndarray): 3D umpy array of N-bit values.
         header (dict): Dictionary of image metadata.
    """
    if '.tif' in path:
        return imread(path)
    elif '.nrrd' in path:
        array, metadata = nrrd.read(path)
        array = np.swapaxes(array, 0, 2)
        if return_header:
            return array, metadata
        else:
            return array


def save_stack(path, stack, header=None, dtype='uint16'):
    """
    Saves a 3D microscopy stack in 8-bit or 16-bit .tif or .nrrd format. Numerical values must be scaled properly
    before using this function (for instance, 16-bit data must be scaled in range [0, 65535]).

    Arguments:
        path (str): Full absolute path of the .tif or .nrrd file to be saved.
        stack (np.ndarray): Numpy array of microscopy data to be saved.
        header (dict): Dictionary of metadata for .nrrd files, optional.
        dtype (str): Numerical encoding of the stack. Works with 'uint8', 'uint16', and probably others.

    Returns:
         Saves the stack at the given path.
    """
    if '.tif' in path:
        imwrite(path, stack.astype(dtype))
    elif '.nrrd' in path:
        if header is None:
            nrrd.write(path, np.transpose(stack.astype(dtype), (2, 1, 0)))
        else:
            nrrd.write(path, np.transpose(stack.astype(dtype), (2, 1, 0)), header)


def load_volumes(directory, frame_ids, keywords=None, exclude=None, plane_marker='#', start_id=2):
    if keywords is None:
        keywords = []
    keywords += ['.tif']
    files = identify_files(directory, keywords=keywords, exclude=exclude)
    N_planes = len(files)
    N = np.linspace(start_id, N_planes + start_id - 1, N_planes).astype('int')
    sorted_files = []
    for n in N:
        for file in files:
            if (plane_marker + '{}_'.format(n) in file) or (plane_marker + '{}.'.format(n) in file):
                sorted_files.append(file)
    frames_full = []
    for f in range(len(files)):
        frames_single = []
        for i in frame_ids:
            frames_single.append(imread(directory + sorted_files[f], key=i))
        frames_full.append(np.stack(frames_single, axis=0))

    return np.stack(frames_full, axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Functions associated with time series analysis / processing
# ----------------------------------------------------------------------------------------------------------------------


def normalize(array):
    normalized = np.copy(array)
    if np.any(array):
        normalized -= np.amin(normalized)
        normalized /= np.amax(normalized)
    return normalized


def baseline_minfilter(signal, window=300, sigma1=5, sigma2=100, debug=False):
    signal_flatstart = np.copy(signal)
    signal_flatstart[0] = signal[1]
    smooth = gaussian_filter1d(signal_flatstart, sigma1)
    mins = minimum_filter1d(smooth, window)
    baseline = gaussian_filter1d(mins, sigma2)
    if debug:
        debug_out = np.asarray([smooth, mins, baseline])
        return debug_out
    else:
        return baseline


def compute_dff_using_minfilter(timeseries, window=200, sigma1=0.1, sigma2=50):
    if len(timeseries.shape) == 1:
        baseline = baseline_minfilter(timeseries, window=window, sigma1=sigma1, sigma2=sigma2)
        dff = (timeseries - baseline) / baseline
    else:
        dff = np.zeros(timeseries.shape)
        for i in range(timeseries.shape[0]):
            if np.any(timeseries[i]):
                baseline = baseline_minfilter(timeseries[i], window=window, sigma1=sigma1, sigma2=sigma2)
                dff[i] = (timeseries[i] - baseline) / baseline
    return dff


def filter_timeseries(timeseries, sigma=2):
    filtered = np.zeros(timeseries.shape)
    for i in range(timeseries.shape[0]):
        filtered[i] = gaussian_filter1d(timeseries[i], sigma)
    return filtered


# ----------------------------------------------------------------------------------------------------------------------
# Classes associated with the mapzebrain atlas and the morphology dataset
# ----------------------------------------------------------------------------------------------------------------------


class Mapzebrain:
    """
    Class to handle Mapzebrain region masks and its coordinate system.
    """

    path = None

    def __init__(self, path=None):

        if path is not None:
            self.path = path

        self.shape = (359, 974, 597)
        self.xlim, self.ylim, self.zlim = [0, 597], [0, 974], [0, 359]
        self.midline = 284
        self.anteroposterior_order = None

        self.region_names = self.load_names()
        self.region_acronyms = self.load_acronyms()
        self.region_masks = self.load_masks()
        self.region_centroids = self.load_centroids()
        self.region_volumes = self.load_volumes()

        self.stack = imread(path + 'H2B-GCaMP6s.tif')
        self.XYprojection = np.mean(self.stack, axis=0)
        self.YZprojection = np.rot90(np.mean(self.stack, axis=2), k=-1)

    def load_volumes(self):
        """Tries to load a 'region_volumes.npy' file in the atlas folder. Returns None if the file doesn't exist."""
        try:
            volumes = np.load(self.path + 'region_volumes.npy')
        except:
            volumes = None
        return volumes

    def load_centroids(self):
        """Tries to load a 'region_centroids.npy' file in the atlas folder. Returns None if the file doesn't exist."""
        try:
            centroids = np.load(self.path + 'region_centroids.npy')
        except:
            centroids = None
        return centroids

    def load_masks(self):
        """Tries to load a 'region_masks.npz' file in the atlas folder. Returns None if the file doesn't exist."""
        try:
            masks = load_npz(self.path + 'region_masks.npz')
        except:
            masks = None
        return masks

    def load_names(self):
        """Tries to load a 'region_names.npy' file in the atlas folder. Returns None if the file doesn't exist."""
        try:
            names = np.load(self.path + 'region_names.npy')
        except:
            names = None
        return names

    def load_acronyms(self):
        """Tries to load a 'region_acronyms.txt' file in the atlas folder. Returns None if the file doesn't exist.
        Acronyms have to be written manually into a .txt file. Verify that the region names and acronyms are in the
        same order."""
        if np.any(identify_files(self.path, ['region_acronyms.txt'])):
            file = open(self.path + 'region_acronyms.txt', 'r')
            lines = file.readlines()
            file.close()
            acronyms = []
            for line in lines:
                acronyms.append(line.split('\n')[0])
            return acronyms
        else:
            return None

    def compile_region_masks(self, folder):
        """
        Loads all .tif region masks contained into a single folder, downloaded from mapZebrain, and compresses them into
        a single memory-efficient sparse array.
        Warning: Make sure that the folder only contains binary region masks, as all .tif files will be loaded
        undiscriminately.

        Arguments:
            folder (str): Full path to a directory containing an ensemble of .tif files.

        Returns:
             masks (sparse): An PxN (pixels x number of regions) sparse array in column format where each column
             is a flattened 3D binary stack representing a single brain region.
        """
        files = identify_files(folder, ['.tif'])
        for i in tqdm(range(len(files))):
            if self.anteroposterior_order is not None:
                mask = load_stack(folder + files[self.anteroposterior_order[i]])
            else:
                mask = load_stack(folder + files[i])
            vector = np.expand_dims(mask.flatten(), axis=1)
            sparse_vector = csc_matrix(vector)
            if i == 0:
                sparse_matrix = sparse_vector
            else:
                sparse_matrix = hstack([sparse_matrix, sparse_vector])
        self.region_masks = sparse_matrix

    def establish_anteroposterior_order(self):
        """
        Computes centers of mass of each brain region to find their ordering along the antero-posterior axis of the
        brain. Changes the 'anteroposterior_order' attribute.
        """
        y_positions = []
        for i in tqdm(range(self.region_masks.shape[1])):
            mask = self.get_region_mask(i)
            _, y, _ = np.where(mask > 0)
            y_positions.append(np.mean(y))
        self.anteroposterior_order = np.argsort(y_positions)

    def compile_region_names(self, folder):
        """
        Infers region names for the names of the .tif files contained in a single folder. Assumes that the .tif files
        are named properly and cleanly.

        Arguments:
            folder (str): Full path to a directory containing an ensemble of .tif files.
        """
        region_names = []
        files = identify_files(folder, ['.tif'])
        for i in range(len(files)):
            if self.anteroposterior_order is not None:
                file = files[self.anteroposterior_order[i]]
            else:
                file = files[i]
            region_names.append(file.split('.tif')[0])
        self.region_names = region_names

    def save_region_masks(self, filename=None):
        """Saves current region masks to .npz sparse format into the main atlas folder."""
        if filename is None:
            save_npz(self.path + 'region_masks.npz', self.region_masks)
        else:
            save_npz(self.path + filename, self.region_masks)

    def save_region_names(self, filename=None):
        """Saves current region names to .npy array into the main atlas folder."""
        if filename is None:
            np.save(self.path + 'region_names.npy', self.region_names)
        else:
            np.save(self.path + filename, self.region_names)

    def get_region_mask(self, region_id):
        """
        Uncompresses and reshapes a single 3D binary stack from the sparse regions matrix.

        Arguments:
            region_id (str): Identifying number of the region to be uncompressed.

        Returns:
            mask (np.ndarray): A 3D binary numpy array shaped according to brain atlas dimensions, where nonzero values
            denote voxels that belong to a certain brain region.
        """
        mask = np.flip(np.reshape(self.region_masks[:, region_id].toarray(), self.shape, order='C'), axis=0)
        return mask

    def generate_masks_projection(self, view='top'):
        """
        Projects all region masks into a single 2D image for visualization.

        Arguments:
            view (str): 'top' or 'side', determines the perspective of the projection.

        Returns:
            projection (np.ndarray): A 2D numpy array depicting all regional masks overlapped.
        """
        R = self.region_masks.shape[1]
        if view == 'top':
            masks = np.zeros((R, self.ylim[1], self.xlim[1]))
            for i in tqdm(range(R)):
                mask = self.get_region_mask(i)
                masks[i] = np.mean(mask, axis=0)
        elif view == 'side':
            masks = np.zeros((R, self.zlim[1], self.ylim[1]))
            for i in tqdm(range(R)):
                mask = self.get_region_mask(i)
                masks[i] = np.mean(mask, axis=2)
        N_intersects = np.sum(masks > 0, axis=0)
        projection = np.mean(masks, axis=0) / (N_intersects + 1)
        return projection

    def map_centroids(self, centroids):
        """
        Maps an array of centroid coordinates into all brain regions.

        Arguments:
            centroids (np.ndarray): Nx3 array of 3D coordinates (in pixels) of N neurons to be mapped in brain regions.
            Must be ordered as (x, y, z).

        Returns:
            region_labels (np.ndarray): NxR boolean numpy array (N neurons x R regions) where True indicates that neuron
            i belongs to brain region j. Summing along columns yields the number of neurons per brain region.
        """
        centroids = self.trim_centroids(centroids)
        region_labels = np.zeros((centroids.shape[0], len(self.region_names)))
        for i in tqdm(range(len(self.region_names))):
            mask = self.get_region_mask(i)
            region_labels[:, i] = mask[centroids[:, 2], centroids[:, 1], centroids[:, 0]]
        region_labels = region_labels.astype('bool')
        return region_labels

    def map_centroids_bihemispheric(self, centroids):
        """
        Similar to the previous method, but doubles the number of brain regions by mapping into separate hemispheres.
        Regions 1 to R correspond to the left hemisphere, while regions R+1 to 2R correspond to the right hemisphere.
        """
        centroids = self.trim_centroids(centroids)
        region_labels = {'left': np.zeros((centroids.shape[0], len(self.region_names))),
                         'right': np.zeros((centroids.shape[0], len(self.region_names)))}
        for i in tqdm(range(len(self.region_names))):
            mask = self.get_region_mask(i)
            for hemisphere in ['left', 'right']:
                half_mask = np.copy(mask)
                if hemisphere == 'left':
                    half_mask[:, :, self.midline:] = 0
                elif hemisphere == 'right':
                    half_mask[:, :, :self.midline] = 0
                region_labels[hemisphere][:, i] = half_mask[centroids[:, 2], centroids[:, 1], centroids[:, 0]]
        region_labels['left'] = region_labels['left'].astype('bool')
        region_labels['right'] = region_labels['right'].astype('bool')
        return np.concatenate([region_labels['left'], region_labels['right']], axis=1)

    def compute_interregional_distances(self):
        """Computes distances (in pixels, but roughly microns) between brain region centroids. Uses both left and right
        hemispheres to compute a representative distance by averaging both inter- and intra-hemispheric distances."""
        N = self.region_masks.shape[1]
        distance = np.zeros((N, N))
        left = self.region_centroids[:N]
        right = self.region_centroids[N:]
        for i in range(N):
            for j in range(i + 1, N):
                distance_max = np.sqrt(
                    (left[i, 0] - right[j, 0]) ** 2 + (left[i, 1] - right[j, 1]) ** 2 + (left[i, 2] - right[j, 2]) ** 2)
                distance_min = np.sqrt(
                    (left[i, 0] - left[j, 0]) ** 2 + (left[i, 1] - left[j, 1]) ** 2 + (left[i, 2] - left[j, 2]) ** 2)
                distance[i, j] = (distance_max + distance_min) / 2
                distance[j, i] = distance[i, j]
        return distance

    def trim_centroids(self, centroids):
        """Clips an Nx3 array of N centroids so that any coordinates that exceed the atlas volume are brought to the
        volume boundaries. Useful to avoid indexing errors when mapping cells in brain regions."""
        centroids[:, 0] = np.clip(centroids[:, 0], 0, self.xlim[1] - 1)
        centroids[:, 1] = np.clip(centroids[:, 1], 0, self.ylim[1] - 1)
        centroids[:, 2] = np.clip(centroids[:, 2], 0, self.zlim[1] - 1)
        return centroids

    def compute_region_centroids(self, align_nodes=True):
        """Computes the 3D centroids each region for both hemispheres. Centroids are compiled into a 2Rx3 array where R
         is the number of brain regions, and columns are ordered as (x, y, z). Centroids 1 to R correspond to the left
         hemisphere, while centroids R+1 to 2R correspond to the right hemisphere."""

        centroids = {}
        centroids['left'], centroids['right'] = np.zeros((self.region_masks.shape[1], 3)), np.zeros((self.region_masks.shape[1], 3))
        for i in tqdm(range(self.region_masks.shape[1])):
            mask = self.get_region_mask(i)
            for hemisphere in ['left', 'right']:
                half_mask = np.copy(mask)
                if hemisphere == 'left':
                    half_mask[:, :, self.midline:] = 0
                elif hemisphere == 'right':
                    half_mask[:, :, :self.midline] = 0
                centroids[hemisphere][i, :] = center_of_mass(half_mask)
        centroids['left'] = np.flip(centroids['left'], axis=1)
        centroids['right'] = np.flip(centroids['right'], axis=1)
        if align_nodes:
            centroids['left'][:, 1] = centroids['right'][:, 1]
            centroids['left'][:, 2] = centroids['right'][:, 2]
        centroids = np.concatenate([centroids['left'], centroids['right']], axis=0)
        self.region_centroids = centroids

    def save_region_centroids(self, filename=None):
        """Saves current region centroids to .npy array into the main atlas folder."""
        if filename is None:
            np.save(self.path + 'region_centroids.npy', self.region_centroids)
        else:
            np.save(self.path + filename, self.region_centroids)

    def compute_region_volumes(self):
        """Computes the fractional (relative) volume of each region."""
        volumes = []
        for i in tqdm(range(self.region_masks.shape[1])):
            mask = self.get_region_mask(i) > 0
            volumes.append(np.sum(mask))
        volumes = np.array(volumes) / np.sum(volumes)
        self.region_volumes = volumes

    def save_region_volumes(self, filename=None):
        """Saves current region volumes to .npy array into the main atlas folder."""
        if filename is None:
            np.save(self.path + 'region_volumes.npy', self.region_volumes)
        else:
            np.save(self.path + filename, self.region_volumes)

    def compile(self, path_masks):
        """Generates all the required files to operate the brain atlas class from a single folder of .tif brain region
        masks. Masks are loaded from 'path_masks', and multiple files are saved in the atlas folder (self.path)."""

        # Region masks
        self.compile_region_masks(path_masks)
        self.establish_anteroposterior_order()
        self.compile_region_masks(path_masks)
        self.save_region_masks()

        # Region names
        self.compile_region_names(path_masks)
        self.save_region_names()

        # Region centroids
        self.compute_region_centroids()
        self.save_region_centroids()

        # Region volumes
        self.compute_region_volumes()
        self.save_region_volumes()

    def compute_centroid_density(self, centroids, sigma=2):
        centroids_trimmed = np.round(self.trim_centroids(centroids)).astype('int')
        density = np.zeros(self.shape)
        for c in np.round(centroids_trimmed).astype('int'):
            density[c[2], c[1], c[0]] += 1
        density = gaussian_filter(density, sigma)
        density_values = density[centroids_trimmed[:, 2], centroids_trimmed[:, 1], centroids_trimmed[:, 0]]
        return density_values

    def generate_centroid_density_stack(self, centroids, sigma=0):
        density = np.zeros(self.shape)
        density[centroids[:, 2], centroids[:, 1], centroids[:, 0]] = 1
        if sigma > 0:
            density = gaussian_filter(density, sigma)
        density = density.astype('double')
        density -= np.amin(density)
        density /= np.amax(density)
        density *= 255
        return density.astype('uint8')


class Neurons:
    """Class to handle the single-neuron reconstructions from mapZebrain. Takes a Mapzebrain object
    as input, and assumes that there is a 'Neurons/' folder within the Mapzebrain folder that contains
    all .swc traced neurons."""

    def __init__(self, atlas):

        self.atlas = atlas
        self.path = atlas.path + 'Neurons/'

        self.neurons = []
        somas, terminals, coords, parents = [], [], [], []

        files = identify_files(self.path)
        for i in range(len(files)):
            data = self.read_swc(self.path + files[i])
            soma = self.get_soma_xyz(data)
            if soma is not None:
                somas.append(soma)
                terminals.append(self.get_terminals_xyz(data))
                c, p = self.get_axon_coordinates(data)
                coords.append(c)
                parents.append(p)
                self.neurons.append(data)

        somas = np.array(somas)
        ids = []
        for i in range(len(terminals)):
            ids += [i] * terminals[i].shape[0]
        ids_coords = []
        for i in range(len(coords)):
            ids_coords += [i] * coords[i].shape[0]

        self.somas = np.stack(somas)
        self.terminals = np.concatenate(terminals)
        self.coords = np.concatenate(coords)
        self.parents = np.concatenate(parents)
        self.ids_terminals = np.array(ids)
        self.ids_coords = np.array(ids_coords)

        self.raw_connectivity = None
        self.W_directed, self.W_undirected = None, None

    def trim_centroids(self, centroids):
        """Clips an Nx3 array of N centroids so that any coordinates that exceed the atlas volume are brought to the
        volume boundaries. Useful to avoid indexing errors when mapping cells in brain regions."""
        centroids[:, 0] = np.clip(centroids[:, 0], 0, self.atlas.xlim[1] - 1)
        centroids[:, 1] = np.clip(centroids[:, 1], 0, self.atlas.ylim[1] - 1)
        centroids[:, 2] = np.clip(centroids[:, 2], 0, self.atlas.zlim[1] - 1)
        return centroids

    def plot_top(self, ax, neuron_id, color=[0, 1, 0], terminals=True, alpha=0.25, soma_size=30, terminal_size=5, linewidth=1, rasterized=False):
        neuron_data = self.neurons[neuron_id]
        soma_xyz = self.get_soma_xyz(neuron_data)
        terminals_xyz = self.get_terminals_xyz(neuron_data)
        ids = []
        for d in neuron_data:
            ids.append(d['id'])
        ids = np.array(ids)
        positions = []
        for i, d in enumerate(neuron_data[1:]):
            try:
                parent = d['parent']
                d_parent = neuron_data[np.where(ids == parent)[0][0]]
                x1, y1 = d_parent['x'], d_parent['y']
                x2, y2 = d['x'], d['y']
                positions.append([(x1, y1), (x2, y2)])
            except:
                pass
        collection = LineCollection(
            positions,
            color=color,
            linewidths=linewidth,
            antialiaseds=(1,),
            alpha=alpha,
            rasterized=rasterized
        )
        ax.add_collection(collection)
        ax.scatter(soma_xyz[0], soma_xyz[1], s=soma_size, color=color, zorder=20, edgecolor='None', alpha=alpha, rasterized=rasterized)
        if terminals:
            ax.scatter(terminals_xyz[:, 0], terminals_xyz[:, 1], color=color, alpha=alpha, s=terminal_size,
                       edgecolor='None', zorder=10, rasterized=rasterized)

    def plot_side(self, ax, neuron_id, color=[0, 1, 0], terminals=True, alpha=0.25, soma_size=30, terminal_size=5, linewidth=1, rasterized=True):
        neuron_data = self.neurons[neuron_id]
        soma_xyz = self.get_soma_xyz(neuron_data)
        terminals_xyz = self.get_terminals_xyz(neuron_data)
        ids = []
        for d in neuron_data:
            ids.append(d['id'])
        ids = np.array(ids)
        positions = []
        for i, d in enumerate(neuron_data[1:]):
            try:
                parent = d['parent']
                d_parent = neuron_data[np.where(ids == parent)[0][0]]
                y1, z1 = d_parent['y'], d_parent['z']
                y2, z2 = d['y'], d['z']
                positions.append([(self.atlas.zlim[-1] - z1, y1), (self.atlas.zlim[-1] - z2, y2)])
            except:
                pass
        collection = LineCollection(
            positions,
            color=color,
            linewidths=linewidth,
            antialiaseds=(1,),
            alpha=alpha,
            rasterized=rasterized
        )
        ax.add_collection(collection)
        ax.scatter(self.atlas.zlim[-1] - soma_xyz[2], soma_xyz[1], s=soma_size, color=color, zorder=20, edgecolor='None', alpha=alpha, rasterized=rasterized)
        if terminals:
            ax.scatter(self.atlas.zlim[-1] - terminals_xyz[:, 2], terminals_xyz[:, 1], color=color, alpha=alpha, s=terminal_size,
                       edgecolor='None', zorder=10, rasterized=rasterized)

    def map_centroids_expanded(self, centroids, boundary_size=30, keep_inside=False):
        centroids = self.trim_centroids(centroids)
        region_labels = np.zeros((centroids.shape[0], len(self.atlas.region_names)))
        for i in tqdm(range(len(self.atlas.region_names)), file=sys.stdout):
            mask = self.atlas.get_region_mask(i)
            dists = distance_transform_edt(1 - (mask / 255))
            if keep_inside:
                mask = (dists <= boundary_size)
            else:
                mask = (dists <= boundary_size) & (dists != 0)
            region_labels[:, i] = mask[centroids[:, 2], centroids[:, 1], centroids[:, 0]]
        region_labels = region_labels.astype('bool')
        return region_labels

    def compute_connectivity(self):
        self.compile_raw_connectivity()
        self.W_directed = self.compute_directed_connectome(self.raw_connectivity)
        self.W_undirected = self.compute_undirected_connectome(self.raw_connectivity)
        print('Done!')

    def compile_raw_connectivity(self):
        connectivity = np.zeros((self.atlas.region_masks.shape[1], self.atlas.region_masks.shape[1]))
        print('Mapping terminals...')
        regions_terminals = self.atlas.map_centroids(np.round(self.terminals).astype('int'))
        print('Mapping somas...')
        regions_somas = self.atlas.map_centroids(np.round(self.somas).astype('int'))
        print('Compiling connections...')
        for i in range(self.terminals.shape[0]):
            if (np.sum(regions_somas[self.ids_terminals[i], :]) > 0) and (np.sum(regions_terminals[i, :]) > 0):
                r1 = np.where(regions_somas[self.ids_terminals[i]])[0][0]
                r2 = np.where(regions_terminals[i])[0][0]
                connectivity[r2, r1] += 1
        self.raw_connectivity = connectivity

    def compute_directed_connectome(self, raw_counts):
        adjacency = np.copy(raw_counts)
        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[0]):
                adjacency[i, j] = adjacency[i, j] / (self.atlas.region_volumes[i] + self.atlas.region_volumes[j])
                adjacency[j, i] = adjacency[j, i] / (self.atlas.region_volumes[i] + self.atlas.region_volumes[j])
        adjacency[adjacency > 0] = np.log10(adjacency[adjacency > 0])
        adjacency[np.diag_indices(adjacency.shape[0])] = 0
        return normalize(adjacency)

    def compute_undirected_connectome(self, raw_counts):
        adjacency = np.copy(raw_counts)
        for i in range(adjacency.shape[0]):
            for j in range(i + 1, adjacency.shape[0]):
                adjacency[i, j] = (adjacency[i, j] + adjacency[j, i]) / (
                        self.atlas.region_volumes[i] + self.atlas.region_volumes[j])
                adjacency[j, i] = adjacency[i, j]
        adjacency[adjacency > 0] = np.log10(adjacency[adjacency > 0])
        adjacency[np.diag_indices(adjacency.shape[0])] = 0
        return normalize(adjacency)

    @staticmethod
    def read_swc(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data_lines = [line for line in lines if not line.startswith('#')]
        neuron_data = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) == 7:
                neuron_data.append({
                    'id': int(parts[0]),
                    'type': int(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': 359.0 - float(parts[4]),
                    'radius': float(parts[5]),
                    'parent': int(parts[6])
                })
        return neuron_data

    @staticmethod
    def get_axon_coordinates(neuron_data):
        coords, parents = [], []
        for d in neuron_data:
            coords.append([d['x'], d['y'], d['z']])
            parents.append(d['parent'])
        return np.stack(coords), np.array(parents)

    @staticmethod
    def get_soma_xyz(neuron_data):
        for d in neuron_data:
            if d['parent'] == -1:
                return d['x'], d['y'], d['z']

    @staticmethod
    def get_terminals_xyz(neuron_data):
        ids = []
        for d in neuron_data:
            ids.append(d['id'])
        ids = np.array(ids)
        is_terminal = np.array([True] * len(ids))
        for d in neuron_data:
            parent = d['parent']
            is_terminal[ids == parent] = False
        centroids = []
        for i, v in enumerate(is_terminal):
            if v:
                d = neuron_data[i]
                centroids.append([d['x'], d['y'], d['z']])
        return np.array(centroids)
