import numpy as np
from scipy import sparse

FILE_NAME = "web-Google.txt"
BETA = 0.85
POWER_ITERATIONS = 100
NUMBER_NODES, EDGES = 875713, 5105039
EPS = 0.0000001


def pagerank(file_name=FILE_NAME, beta=BETA, power_iterations=POWER_ITERATIONS, num_nodes=NUMBER_NODES,
             num_edges=EDGES, eps=EPS):
    """Simple pageRank using power iteration method"""

    with open(file_name, "r") as f:
        content = f.read().split("\n")
        content = [x.split("\t") for x in content][4:-1]

        # All nodes have ids, which do not correspond to their order of appearance
        # thus, hash these nodes from 0 to num_nodes
        nodes, id = set(int(x) for l in content for x in l), 0
        nodes_dict = dict()
        for unique_node in nodes:
            nodes_dict[unique_node] = id
            id += 1

        row, col = [], []
        for tuple_ in content:
            x_coord = nodes_dict[int(tuple_[0])]
            y_coord = nodes_dict[int(tuple_[1])]
            row.append(y_coord), col.append(x_coord)

    matrix = sparse.csr_matrix(([True] * num_edges, (row, col)), shape=(num_nodes, num_nodes))
    print("M matrix is formed, shape: ", matrix.shape)

    # A = BM + (1 - B) / N
    array_r = np.ones((num_nodes, 1)) / num_nodes
    new_array_r = np.ones((num_nodes, 1))

    # POWER ITERATION
    degree = matrix.sum(axis=0).T
    num_i = 0
    while num_i < power_iterations:
        new_array_r = matrix.dot(beta * (array_r / degree))
        new_array_r += (1 - new_array_r.sum()) / num_nodes

        dist = np.linalg.norm(array_r - new_array_r)  # distance metric is optional
        if dist < eps:
            break
        array_r = new_array_r
        num_i += 1
        print("Iteration : ", num_i, ' score:', dist)

    print("Result:", array_r)
    print("Min rank is for page with id:",list(nodes_dict.keys())[list(nodes_dict.values()).index(np.argmin(array_r))])
    print("Max rank is for page with id:",list(nodes_dict.keys())[list(nodes_dict.values()).index(np.argmax(array_r))])

    return array_r

# pagerank()
