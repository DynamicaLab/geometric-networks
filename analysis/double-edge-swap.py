import numpy as np



def double_edge_swap(adjacency_matrix, number_of_iterations=None, preserve_out_strength=False, distance_matrix=None, min_distance=None, max_distance=None):

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
    else:
        edgelist = np.argwhere(adjacency_matrix)
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
        if weighted:
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
            if undirected:
                if np.random.rand() < 0.5:
                    weight1, weight2 = weight2, weight1
            else:
                if preserve_out_strength:
                    weight1, weight2 = weight2, weight1
    
        adjacency_matrix[target1, source2] = weight1
        adjacency_matrix[target2, source1] = weight2
        if undirected:
            adjacency_matrix[source2, target1] = weight1
            adjacency_matrix[source1, target2] = weight2

        edgelist[e1, 0], edgelist[e1, 1] = target1, source2
        edgelist[e2, 0], edgelist[e2, 1] = target2, source1
    
    return adjacency_matrix


def within_distance_interval(target1, source1, target2, source2, distance_matrix, min_distance, max_distance):
    if distance_matrix != None:
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