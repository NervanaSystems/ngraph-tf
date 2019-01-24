# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import tensorflow as tf
from google.protobuf import text_format
import sys, json, os, numpy as np

def model_reader(graph_filename):
    if not tf.gfile.Exists(graph_filename):
        raise Exception("Input graph file '" + graph_filename +
                        "' does not exist!")

    graph_def = tf.GraphDef()
    if graph_filename.endswith("pbtxt"):
        with open(graph_filename, "r") as f:
            text_format.Merge(f.read(), graph_def)
    else:
        with open(graph_filename, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def

def parse_json(config_file):
    with open(os.path.abspath(config_file)) as f:
        return json.load(f)

def run_single_input(graph_def, output_tensor_names, name_data_map):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    feed_dict = {graph.get_tensor_by_name(inname):name_data_map[inname] for inname in name_data_map}

    config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=1)
    with tf.Session(graph=graph, config=config) as sess:
        output_tensor = sess.run([graph.get_tensor_by_name(outname) for outname in output_tensor_names], feed_dict=feed_dict)
        return {name:val for name,val in zip(output_tensor_names, output_tensor)}

def run(graph_def, output_tensor_names, input_folder_name, input_tensor_name):
    for flname in os.listdir(input_folder_name):
        fullpath = input_folder_name.rstrip('/') + '/' + flname
        if '.bin' in flname and os.path.isfile(fullpath):
            out_results = run_single_input(graph_def, output_tensor_names, {input_tensor_name: np.fromfile(fullpath, dtype=np.float32).reshape([1, 224,224,3])})
            for tname in out_results:
                with open(flname.split('.bin')[0] + "___" + tname.replace(':', '-').replace('/', '_') + '.bin', 'wb') as f:
                    f.write(out_results[tname].tobytes())

if __name__ == '__main__':
    json_data = parse_json(sys.argv[1])
    graph_def = model_reader(json_data['graph_location'])
    run(graph_def, json_data['output_tensor_name'], json_data['input_loc'], json_data['input_tensor_name'])

# Run command:
# python tf_dump.py config.json


