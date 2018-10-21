from __future__ import print_function
import tensorflow as tf, pdb
from ngtf_graph_viewer import load_file

def create_poset_grouping(poset, sink_first=False):
    """Given a partial ordering map, returns uncomparable element groupings.

    Args:
        poset (dictionary<T, list<T>>): Can be source first, eg: {0: [1, 2,3], 1: [4,5], 2: [4,6], 3: [5,6], 4:[7], 5:[7], 6:[7]}
        Can be sink first: {7: [4,5,6], 4: [1,2], 5: [1,3], 6: [2,3], 1: [0], 2: [0], 3: [0]}
        sink_first (Bool or None): Indicates if poset is in source-first or sink-first representation

    Returns:
        dictionary<int, set<T>>: Groups non-comparables together. Keys values indicate sort order.

    Example:
        # Poset on the relation 'is subset' in the powerset of a 3 element set
        poset = {0: [1,2,3], 1: [4,5], 2: [4,6], 3: [5,6], 4:[7], 5:[7], 6:[7]}
        poset1 = {7: [4,5,6], 4: [1,2], 5: [1,3], 6: [2,3], 1: [0], 2: [0], 3: [0]}
        print create_poset_grouping(poset)  # {0: set([0]), 1: set([1, 2, 3]), 2: set([4, 5, 6]), 3: set([7])}
        print create_poset_grouping(poset1, True) #{0: set([0]), 1: set([1, 2, 3]), 2: set([4, 5, 6]), 3: set([7])}
    """
    # TODO: check_posetness(ordering)

    def pop_endpoints_in_poset(poset):
        endpoints = set(poset.keys()).difference(set(sum(poset.values(), [])))
        for i in endpoints:
            del poset[i]
        return endpoints

    c = 0
    groupings = {}
    extreme_endpoints = set(sum(poset.values(), [])).difference(set(poset.keys()))
    while len(poset)!=0:
        groupings[c] = pop_endpoints_in_poset(poset)
        c += 1
    groupings[c] = extreme_endpoints

    # The algorithm is agnostic to source-first or sink-first representation up to this point
    if sink_first:
        return {c-k:groupings[k] for k in groupings}
    else:
        return groupings

def topo_sort_graph(graph):
    # inp_nodes_gdef = filter(lambda n : len(n.input)==0 , graphdef.node)
    # inp_node_name_size_map = {k.name : [d.size for d in k.attr["shape"].shape.dim] for k in inp_nodes}

    inp_ops = filter(lambda n : len(n.inputs._inputs)==0, graph.get_operations())
    op_to_output_edge_info = {}
    # a map whose keys are tensors (edges) and values are tuples indicating src and dest node (operation)
    # tuple[0] is singkle element, tuple[1] is a list. because source is unique, destination can be multiple nodes
    edge_to_srcdst_map = {}  
    for op in graph.get_operations():  #for each operation (or node)
        for out_edge in op.outputs:
            old_srcdest_tuple = edge_to_srcdst_map.get(out_edge, (None, []))
            assert old_srcdest_tuple[0] is None # assert we have not already assigned src of this edge
            edge_to_srcdst_map[out_edge] = (op, old_srcdest_tuple[1])
        for in_edge in op.inputs:
            old_srcdest_tuple = edge_to_srcdst_map.get(in_edge, (None, []))
            edge_to_srcdst_map[in_edge] = (old_srcdest_tuple[0], old_srcdest_tuple[1] + [op])

    poset = {}
    for edge in edge_to_srcdst_map:
        src, dests = edge_to_srcdst_map[edge]
        poset[src] = poset.get(src, []) + dests
    grouped = create_poset_grouping(poset, False)  #TODO: not working. debug
    pdb.set_trace()     


#load_file(graph_file_name, input_binary)

def sample_network():
    n_input = 100
    n_classes = 20
    n_hidden_1 = 20
    x = tf.placeholder("float", [None, n_input])
    a = tf.matmul(x, tf.Variable(tf.random_normal([n_input, n_hidden_1])))
    b = tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    layer_1 = tf.add(a, b)
    layer_1 = tf.add(layer_1, b)
    layer_1 = tf.nn.relu(layer_1)
    return tf.get_default_graph()

#in_tensor_list = [tf.get_default_graph().get_tensor_by_name(tname) for tname in input_tensor_name_list]

topo_sort_graph(sample_network())








