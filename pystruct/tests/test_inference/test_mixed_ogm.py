import numpy as np
from numpy.testing.utils import assert_array_equal
from pystruct.inference import inference_dispatch

__author__ = 'tolgab'


def test_tree():
    # build tree
    n_nodes = 6
    n_states_per_node = [3, 3, 2, 2, 2, 2]
    # make edges
    edges = np.array([[0, 2], [1, 2], [2, 3], [2, 4], [3, 5]], dtype=int)
    n_edges = len(edges)
    # these have high unary pots, the others have 0 unary pots
    correct_states = [0, 0, 1, 1, 0, 1]
    # fill unary pots
    unary_pots = np.zeros(n_nodes, dtype=object)
    for i in range(n_nodes):
        unary_pots[i] = np.zeros(n_states_per_node[i], dtype=float)
        unary_pots[i][correct_states[i]] = 10.
    # fill pairwise pots
    pw_pots = np.zeros(n_edges, dtype=object)
    for i in range(n_edges):
        pw_pots[i] = np.zeros((n_states_per_node[edges[i][0]],
                               n_states_per_node[edges[i][1]]), dtype=float)
        pw_pots[i][correct_states[edges[i][0]], correct_states[edges[i][1]]] = 15.
    res = inference_dispatch(unary_pots, pw_pots, edges, "mixed_ogm")
    assert_array_equal(res,correct_states)

if __name__ == "__main__":
    test_tree()