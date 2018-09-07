from __future__ import print_function

import ngraph
import tensorflow as tf, numpy as np, re, os, pdb
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import argparse, pickle as pkl

def modify_node_names(graph_def, node_map):
    for node in graph_def.node:
        if node.name in node_map:
            old_name = node.name
            new_name = node_map.get(node.name)
            print("Replacing: ", node.name, " with ", new_name)
            node.name = new_name
            for node in graph_def.node:
                for idx, inp_name in enumerate(node.input):
                    if inp_name == old_name:
                        node.input[idx] = new_name
                #TODO: Do we need to edit this anywhere else other than inputs?
    return graph_def

# remove '_' from node names
def sanitize_node_names(graph_def):
    return modify_node_names(graph_def, {node.name : node.name[1:] for node in graph_def.node if node.name[0] == "_"})

# prepend an extra string to the node name (presumably a scope, to denote encapsulate)
def prepend_to_name(graph_def, node_map):
    return modify_node_names(graph_def, {node.name : node_map[node.name] + node.name for node in graph_def.node if node.name in node_map})

# can load protobuf (pb or pbtxt). can modify only pbtxt for now
def load_file(graph_file, input_binary, modifier_function_list=[]):
    if not gfile.Exists(graph_file):
        raise Exception("Input graph file '" + graph_file + "' does not exist!")
    if input_binary and len(modifier_function_list) > 0:
        # TODO: can we support modifications on .pb? perhaps by: pb->pbtxt->modify->pb?
        raise Exception("Input of type .pb. Modifications supported only in .pbtxt")

    graphdef = graph_pb2.GraphDef()
    with open(graph_file, "r") as f:
        protobuf_str = f.read()
        try:
            if input_binary:
                graphdef.ParseFromString(protobuf_str)
            else:
                text_format.Merge(protobuf_str, graphdef)
        except:
            raise Exception("Failed to read pb or pbtxt. input_binary is " + str(input_binary) + " maybe try flipping it?")
    for modifier_function in modifier_function_list:
        graphdef = modifier_function(graphdef)
    return graphdef

def preprocess(input_filename, out_dir, input_binary, node_map):
    # Note: node_map should be applied before sanitize_node_names.
    # Else sanitize_node_names might change the node names, which might become unrecognizable to node_map
    modifiers = [] if input_binary else [lambda pbtxt_str : prepend_to_name(pbtxt_str, node_map), sanitize_node_names]
    gdef = load_file(input_filename, input_binary, modifiers)
    if not os.path.exists(out_dir): #create output dir if it does not exist
        os.makedirs(out_dir)
    return gdef

def graphdef_to_dot(gdef, dot_output):
    with open(dot_output, "wb") as f:
        print("digraph graphname {", file=f)
        for node in gdef.node:
            output_name = node.name
            print("  \"" + output_name + "\" [label=\"" + node.op + "\"];", file=f)
            for input_full_name in node.input:
                parts = input_full_name.split(":")
                input_name = re.sub(r"^\^", "", parts[0])
                print("  \"" + input_name + "\" -> \"" + output_name + "\";", file=f)
        print("}", file=f)
    print("\n"+('=-'*30))
    print("Created DOT file '" + dot_output + "'.")
    print("Can be converted to pdf using: dot -Tpdf "+dot_output+" -o "+dot_output+".pdf")
    print('=-'*30)

def protobuf_to_dot(input_filename, dot_dir, input_binary=False, node_map={}):
    gdef = preprocess(input_filename, dot_dir, input_binary, node_map)
    graphdef_to_dot(gdef, dot_dir.rstrip('/') + '/' + os.path.basename(input_filename) + '.dot')

def graphdef_to_tensorboard(gdef, tensorboard_output):
    # convert graphdef to graph, even though FileWriter can accepts graphdefs.
    # this is because FileWriter has deprecated graphdef as inputs, and prefers graphs as inputs
    with tf.Session() as sess:
        tf.import_graph_def(gdef)
        writer = tf.summary.FileWriter(tensorboard_output, sess.graph)
        #TODO: try with tf master
        # wont work now if we have NGraphVariable, NGraphEncapsulateOp
        # TODO: How about supporting NGraphVariable and NGraphEncapsulateOp by switching their optype with something TB knows
        writer.flush()
        writer.close()
    # It seems NGraphVariable and NGraphEncapsulateOp are registered in C++ but not in python
    print("\n"+('=-'*30)+"\nTo view Tensorboard:")
    print("1) Run this command: tensorboard --logdir " + tensorboard_output)
    print("2) Go to the URL it provides or http://localhost:6006/\n"+('=-'*30)+"\n")

def protobuf_to_grouped_tensorboard(input_filename, tensorboard_dir, input_binary=False, node_map={}):
    gdef = preprocess(input_filename, tensorboard_dir, input_binary, node_map)
    graphdef_to_tensorboard(gdef, tensorboard_dir)

visualizations_supported = [protobuf_to_dot, protobuf_to_grouped_tensorboard]

if __name__ == "__main__":
    helptxt = '''
    Convert protobuf to different visualizations (dot, tensorboard).

    Sample usage from command line:
    python protobuf_visualize.py pbtxtfile.pbtxt ./vis  # read pbtxt and generate TB
    python protobuf_visualize.py -v 1 pbtxtfile.pbtxt ./vis  # read pbtxt and generate dot
    python protobuf_visualize.py -b pbtxtfile.pb ./vis  # read pb and generate TB
    python protobuf_visualize.py -b -v 1 pbtxtfile.pb ./vis  # read pb and generate dot
    python protobuf_visualize.py -c nodemap.pkl pbtxtfile.pbtxt ./vis  # read pbtxt, remap node names and generate TB
    One can also import the file and use its functions
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=helptxt)
    parser.add_argument("input", help="The input protobuf (pb or pbtxt)")
    parser.add_argument("out", help="The output directory")
    parser.add_argument('-b', dest='binary', action='store_true', help="Add this flag to indicate its a .pb. Else it is assumed to be a .pbtxt")
    parser.add_argument("-v", "--visualize", type=int, default=1, help="Enter 0 (protobuf->dot) or 1 (protobuf->Tensorboard). By default it converts to tensorboard")
    parser.add_argument("-c", "--cluster", help="An file that contains the node-to-cluster map that can be used to group them into clusters")
    args = parser.parse_args()

    node_map = {} if args.cluster is None else pkl.load(open(args.cluster, 'rb'))
    visualizations_supported[args.visualize](args.input, args.out, args.binary, node_map)
