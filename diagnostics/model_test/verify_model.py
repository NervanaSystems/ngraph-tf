# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import tensorflow as tf
import argparse
import numpy as np
import ngraph_bridge
from google.protobuf import text_format
import json
import os
import sys


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def set_os_env(select_device):
    if select_device == 'CPU':
        # run on TF only
        ngraph_bridge.disable()
    else:
        if not ngraph_bridge.is_enabled():
            ngraph_bridge.enable()
        assert select_device[:
                             7] == "NGRAPH_", "Expecting device name to start with NGRAPH_"
        back_end = select_device.split("NGRAPH_")
        os.environ['NGRAPH_TF_BACKEND'] = back_end[1]


def calculate_output(param_dict, select_device, input_example):
    """Calculate the output of the imported graph given the input.

    Load the graph def from graph file on selected device, then get the tensors based on the input and output name from the graph,
    then feed the input_example to the graph and retrieves the output vector.

    Args:
    param_dict: The dictionary contains all the user-input data in the json file.
    select_device: "NGRAPH" or "CPU".
    input_example: A map with key is the name of the input tensor, and value is the random generated example

    Returns:
        The output vector obtained from running the input_example through the graph.
    """
    tf.reset_default_graph()
    is_ckpt = False

    if "pb_graph_location" in param_dict and "checkpoint_graph_location" in param_dict:
        raise Exception(
            "Only Graph or Checkpoint file can be specified, not both!")

    if "pb_graph_location" in param_dict:
        pb_filename = param_dict["pb_graph_location"]
    elif "checkpoint_graph_location" in param_dict:
        checkpoint_filename = param_dict["checkpoint_graph_location"]
        is_ckpt = True
    else:
        raise Exception(
            "Input graph file OR Input checkpoint file is required!")

    output_tensor_name = param_dict["output_tensor_name"]

    config = tf.ConfigProto(inter_op_parallelism_threads=1)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    set_os_env(select_device)

    # if checkpoint, then load checkpoint
    if (is_ckpt):
        meta_filename = checkpoint_filename + '.meta'
        if not tf.gfile.Exists(meta_filename):
            raise Exception("Meta file does not exist")
        else:
            saver = tf.train.import_meta_graph(meta_filename)

        if not tf.train.checkpoint_exists(checkpoint_filename):
            raise Exception("Checkpoint with this prefix does not exist")
        else:
            saver.restore(sess, checkpoint_filename)

        print("Model restored: " + select_device)
        graph = tf.get_default_graph()

    #if graph, then load graph
    else:
        graph_def = tf.GraphDef()
        if pb_filename.endswith("pbtxt"):
            with open(pb_filename, "r") as f:
                text_format.Merge(f.read(), graph_def)
        else:
            with open(pb_filename, "rb") as f:
                graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)
        sess = tf.Session(graph=graph, config=config)

    # if no outputs are specified, then compare for all tensors
    if len(output_tensor_name) == 0:
        output_tensor_name = sum(
            [[j.name for j in i.outputs] for i in graph.get_operations()], [])

    # Create the tensor to its corresponding example map
    tensor_to_example_map = {}
    for item in input_example:
        t = graph.get_tensor_by_name(item)
        tensor_to_example_map[t] = input_example[item]

    tensors = []
    skipped_tensors = []
    output_tensor = [graph.get_tensor_by_name(i) for i in output_tensor_name]
    for name in output_tensor_name:
        try:
            output_tensor = sess.run(name, feed_dict=tensor_to_example_map)
            tensors.append(output_tensor)
        except Exception as e:
            skipped_tensors.append(name)
    return tensors, output_tensor_name, skipped_tensors


def calculate_norm(ngraph_output, tf_output, desired_norm):
    """Calculate desired_norm between vectors.

    Calculate the L1/L2/inf norm between the NGRAPH and tensorflow output vectors.

    Args:
        ngraph_output: The output vector generated from NGRAPH graph.
        tf_output: The output vector generated from tensorflow graph.
        desired_norm: L1/L2/inf norm. 

    Returns:
        Calculated norm between the vectors.

    Raises:
        Exception: If the dimension of the two vectors mismatch.
    """
    if (ngraph_output.shape != tf_output.shape):
        raise Exception('ngraph output and tf output dimension mismatch')

    ngraph_output_squeezed = np.squeeze(ngraph_output)
    tf_output_squeezed = np.squeeze(tf_output)

    #if size of node is 1 but shape is (), reshaping it to (1,)
    if (len(ngraph_output_squeezed.shape) == 0):
        ngraph_output_squeezed = ngraph_output_squeezed.reshape([1])
        tf_output_squeezed = tf_output_squeezed.reshape([1])

    ngraph_output_flatten = ngraph_output_squeezed.flatten()
    tf_output_flatten = tf_output_squeezed.flatten()

    factor = np.prod(ngraph_output_squeezed.shape)

    if desired_norm not in [1, 2, np.inf]:
        raise Exception('Only L2, L2, and inf norms are supported')

    #Additional check to verify if the op datatype can be converted or not to be able to subtract.
    #Few data types cannot be converted, the list is printed among the results at the end.
    if ngraph_output_flatten.size is not 0:
        try:
            n = np.linalg.norm((ngraph_output_flatten.astype(np.float32) -
                                tf_output_flatten.astype(np.float32)),
                               desired_norm)
        except:
            n = None

        if desired_norm is np.inf or n is None:
            return n
        else:
            return n / len(ngraph_output_flatten)


def parse_json():
    """
        Parse the user input json file.

        Returns:
            A dictionary contains all the parsed parameters.
    """
    with open(os.path.abspath(args.json_file)) as f:
        parsed_json = json.load(f)
        return parsed_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file", type=str, help="Model details in json format")

    args = parser.parse_args()

    if args.json_file is None:
        raise ValueError("Supply a json file to start")

    parameters = parse_json()

    # Get reference/testing backend to compare
    device1 = parameters["reference_backend"]
    device2 = parameters["testing_backend"]

    # Get L1/L2/Inf threshold value
    l1_norm_threshold = parameters["l1_norm_threshold"]
    l2_norm_threshold = parameters["l2_norm_threshold"]
    inf_norm_threshold = parameters["inf_norm_threshold"]

    # Create a folder to save output tensor arrays
    output_folder = device1 + "-" + device2
    createFolder(output_folder)
    os.chdir(output_folder)
    print("Model name: " + parameters["model_name"])
    print("L1/L2/Inf norm configuration: {}, {}, {}".format(
        l1_norm_threshold, l2_norm_threshold, inf_norm_threshold))

    # Generate random input based on input_dimension
    np.random.seed(100)
    input_dimension = parameters["input_dimension"]
    input_tensor_name = parameters["input_tensor_name"]
    # Get random value range
    rand_val_range = parameters["random_val_range"]
    bs = int(parameters["batch_size"])

    assert len(input_dimension) == len(
        input_tensor_name
    ), "input_tensor_name dimension should match input_dimension in json file"

    assert len(input_tensor_name) == len(
        rand_val_range
    ), "Length of random_val_range should match input_tensor_name in json file"

    # Matches the input tensors name with its required dimensions
    input_tensor_dim_map = {}
    for (dim, name, val_range) in zip(input_dimension, input_tensor_name,
                                      rand_val_range):
        random_input = np.random.randint(
            val_range, size=[bs] + dim).astype('float32')
        input_tensor_dim_map[name] = random_input

    # Run the model on reference backend
    result_tf_graph_arrs, out_tensor_names_cpu, tf_skipped_tensors = calculate_output(
        parameters, device1, input_tensor_dim_map)
    # Run the model on testing backend
    result_ngraph_arrs, out_tensor_names_ngraph, ngraph_skipped_tensors = calculate_output(
        parameters, device2, input_tensor_dim_map)

    assert all(
        [i == j for i, j in zip(tf_skipped_tensors, ngraph_skipped_tensors)])

    print("Skipping comparison of the output tensors below:")
    for tensor in tf_skipped_tensors:
        print("\n[" + tensor + "]")

    assert all(
        [i == j for i, j in zip(out_tensor_names_cpu, out_tensor_names_ngraph)])
    passed = True
    th_dict = {
        "L1": l1_norm_threshold,
        "L2": l2_norm_threshold,
        "inf": inf_norm_threshold
    }
    for tname, result_ngraph, result_tf_graph in zip(
            out_tensor_names_cpu, result_ngraph_arrs, result_tf_graph_arrs):
        new_out_layer = tname.replace("/", "_")
        nparray_tf = np.array(result_tf_graph)
        nparray_ngraph = np.array(result_ngraph)
        np.save(device1 + "-" + new_out_layer + ".npy", nparray_tf)
        np.save(device2 + "-" + new_out_layer + ".npy", nparray_ngraph)

        l1_norm = calculate_norm(nparray_ngraph, nparray_tf, 1)
        l2_norm = calculate_norm(nparray_ngraph, nparray_tf, 2)
        inf_norm = calculate_norm(nparray_ngraph, nparray_tf, np.inf)

        norm_dict = {"L1": l1_norm, "L2": l2_norm, "inf": inf_norm}
        print("\n[" + tname + "]")
        #start the loop and check norms
        for norm_name in norm_dict:
            np.set_printoptions(precision=15)
            if norm_dict[norm_name] is None:
                print("Data type conversion failed, so not comparing outputs")
                passed = False
                break
            elif norm_dict[norm_name] > th_dict[norm_name]:
                print(
                    "The %s norm is greater than %s threshold - %s norm: %f, %s threshold: %f"
                    % (norm_name, norm_name, norm_name, norm_dict[norm_name],
                       norm_name, th_dict[norm_name]))
                passed = False
            else:
                print("The %s norm test passed - %s norm: %f, %s threshold: %f"
                      % (norm_name, norm_name, norm_dict[norm_name], norm_name,
                         th_dict[norm_name]))
    if not passed:
        sys.exit(1)
