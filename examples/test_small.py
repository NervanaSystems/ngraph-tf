import tensorflow as tf
import ngraph_bridge
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
import pdb

def get_sess():
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)
    try:
        if ngraph_bridge.is_grappler_enabled():
            rewrite_options = rewriter_config_pb2.RewriterConfig(
                    meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE,
                    custom_optimizers=[
                        rewriter_config_pb2.RewriterConfig.CustomGraphOptimizer(
                            name="ngraph-optimizer")
                    ])
            config.MergeFrom(
                    tf.ConfigProto(
                        graph_options=tf.GraphOptions(rewrite_options=rewrite_options)))
    except:
        pass
    return tf.Session(config=config)

def get_graph(var):
    if var:
        x = tf.Variable(np.full((2, 3), 1.0))
        y = tf.Variable(np.full((2, 3), 1.0))
        z = tf.Variable(np.full((2, 3), 1.0))
    else:
        x = tf.placeholder(tf.float32, shape=(2, 3))
        y = tf.placeholder(tf.float32, shape=(2, 3))
        z = tf.placeholder(tf.float32, shape=(2, 3))
    #pdb.set_trace()

    a = x + y + z
    return [a, tf.nn.sigmoid(a)], [x, y, z]

def run_graph(var):
    outs, ins = get_graph(var)
    sess = get_sess()
    if var:
        sess.run(tf.global_variables_initializer())
    res = sess.run(outs, feed_dict = {} if var else {i:np.full((2, 3), 1.0)  for i in ins})
    print(res)

run_graph(True)
