import numpy as np
import copy


def within_distance_interval(target1: int, source1: int, target2: int, source2: int, distance_matrix: np.ndarray, min_distance: float = None, max_distance: float = None) -> bool:
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


def double_edge_swap(adjacency_matrix: np.ndarray, weights: str = None, number_of_iterations: int = None, distance_matrix: np.ndarray = None, min_distance: float = None, max_distance: float = None) -> np.ndarray:
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
        number_of_iterations = 10*number_of_edges

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
        if distance_matrix != None:
            if not within_distance_interval(target1, source1, target2, source2, distance_matrix, min_distance, max_distance):
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
                raise ValueError("The 'weights' parameter must be either 'random', 'preserve-out-strength', or 'preserve-in-strength'.")
    
        adjacency_matrix[target1, source2] = weight1
        adjacency_matrix[target2, source1] = weight2
        if undirected:
            adjacency_matrix[source2, target1] = weight1
            adjacency_matrix[source1, target2] = weight2

        edgelist[e1, 0], edgelist[e1, 1] = target1, source2
        edgelist[e2, 0], edgelist[e2, 1] = target2, source1
    
    return adjacency_matrix


if __name__ == "__main__":

    import networkx as nx

    number_of_vertices = 100
    average_degree = 2
    number_of_edges = average_degree * number_of_vertices / 2
    A_undirected_unweighted = nx.to_numpy_array(nx.gnm_random_graph(number_of_vertices, number_of_edges, directed=False))
    A_directed_unweighted = nx.to_numpy_array(nx.gnm_random_graph(number_of_vertices, number_of_edges, directed=True))

    average_weight = 5
    variance_weight = 2
    theta = variance_weight / average_weight
    k = average_weight / theta
    A_undirected_weighted = np.multiply(A_undirected_unweighted, np.random.gamma(shape=k, scale=theta, size=(number_of_vertices, number_of_vertices)))
    A_undirected_weighted = (A_undirected_weighted + A_undirected_weighted.T) / 2
    A_directed_weighted = np.multiply(A_directed_unweighted, np.random.gamma(shape=k, scale=theta, size=(number_of_vertices, number_of_vertices)))

    A_undirected_unweighted_shuffled = double_edge_swap(A_undirected_unweighted)
    assert not np.allclose(A_undirected_unweighted, A_undirected_unweighted_shuffled)
    assert np.allclose(A_undirected_unweighted_shuffled, A_undirected_unweighted_shuffled.T)
    assert np.isclose(np.sum(A_undirected_unweighted), np.sum(A_undirected_unweighted_shuffled))
    assert np.allclose(np.sum(A_undirected_unweighted, axis=0), np.sum(A_undirected_unweighted_shuffled, axis=0))
    assert np.allclose(np.sum(A_undirected_unweighted, axis=1), np.sum(A_undirected_unweighted_shuffled, axis=1))

    A_directed_unweighted_shuffled = double_edge_swap(A_directed_unweighted)
    assert not np.allclose(A_directed_unweighted, A_directed_unweighted_shuffled)
    assert not np.allclose(A_directed_unweighted_shuffled, A_directed_unweighted_shuffled.T)
    assert np.isclose(np.sum(A_directed_unweighted), np.sum(A_directed_unweighted_shuffled))
    assert np.allclose(np.sum(A_directed_unweighted, axis=0), np.sum(A_directed_unweighted_shuffled, axis=0))
    assert np.allclose(np.sum(A_directed_unweighted, axis=1), np.sum(A_directed_unweighted_shuffled, axis=1))

    A_undirected_weighted_shuffled = double_edge_swap(A_undirected_weighted)
    assert not np.allclose(A_undirected_weighted, A_undirected_weighted_shuffled)
    assert np.allclose(A_undirected_weighted_shuffled, A_undirected_weighted_shuffled.T)
    assert np.isclose(np.sum(A_undirected_weighted), np.sum(A_undirected_weighted_shuffled))
    assert not np.allclose(np.sum(A_undirected_weighted, axis=0), np.sum(A_undirected_weighted_shuffled, axis=0))
    assert not np.allclose(np.sum(A_undirected_weighted, axis=1), np.sum(A_undirected_weighted_shuffled, axis=1))

    A_directed_weighted_shuffled_out_strength_preserved = double_edge_swap(A_directed_weighted)
    assert not np.allclose(A_directed_weighted, A_directed_weighted_shuffled_out_strength_preserved)
    assert not np.allclose(A_directed_weighted_shuffled_out_strength_preserved, A_directed_weighted_shuffled_out_strength_preserved.T)
    assert np.isclose(np.sum(A_directed_weighted), np.sum(A_directed_weighted_shuffled_out_strength_preserved))
    assert np.allclose(np.sum(A_directed_weighted, axis=0), np.sum(A_directed_weighted_shuffled_out_strength_preserved, axis=0))
    assert not np.allclose(np.sum(A_directed_weighted, axis=1), np.sum(A_directed_weighted_shuffled_out_strength_preserved, axis=1))

    A_directed_weighted_shuffled_in_strength_preserved = double_edge_swap(A_directed_weighted, weights='preserve-in-strength')
    assert not np.allclose(A_directed_weighted, A_directed_weighted_shuffled_in_strength_preserved)
    assert not np.allclose(A_directed_weighted_shuffled_in_strength_preserved, A_directed_weighted_shuffled_in_strength_preserved.T)
    assert np.isclose(np.sum(A_directed_weighted), np.sum(A_directed_weighted_shuffled_in_strength_preserved))
    assert not np.allclose(np.sum(A_directed_weighted, axis=0), np.sum(A_directed_weighted_shuffled_in_strength_preserved, axis=0))
    assert np.allclose(np.sum(A_directed_weighted, axis=1), np.sum(A_directed_weighted_shuffled_in_strength_preserved, axis=1))
