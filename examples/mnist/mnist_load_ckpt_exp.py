import tensorflow as tf
import ngraph_bridge

from tensorflow.examples.tutorials.mnist import input_data


checkpoint_dir = "./mnist_trained/model.ckpt"
data_dir ='/tmp/tensorflow/mnist/input_data'
train_loops = 50
batch_size=50

# Config
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

# Note: Additional configuration option to boost performance is to set the
# following environment for the run:
# OMP_NUM_THREADS=44 KMP_AFFINITY=granularity=fine,scatter
# The OMP_NUM_THREADS number should correspond to the number of
# cores in the system

# Import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

saver = tf.train.import_meta_graph(checkpoint_dir +".meta")

with tf.Session(config=config) as sess:
    # Restore variables from disk.
    saver.restore(sess, checkpoint_dir)
    print("Model restored.")
    op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    op_types = [n.op for n in tf.get_default_graph().as_graph_def().node]
    print("OP Names")
    print (op_names)
    print("OP TYPES")
    print (op_types)
    tf.train.write_graph(sess.graph_def, './', 'reloaded_graph.pbtxt')
    #tf.train.write_graph(sess.graph_def, "/tmp/load", "test.pb", False) #proto
    #writer = tf.summary.FileWriter("./reloaded_graph.pbtxt", sess.graph)
    for i in range(train_loops):
        batch = mnist.train.next_batch(batch_size)
        loss = sess.run(["Mean:0"],
                            feed_dict={
                                "Placeholder:0": batch[0],
                                "Placeholder_3:0": batch[0],
                                "Placeholder_1:0": batch[1],
                                "Placeholder_1_1:0": batch[1]
                            })
        print ("itr ", i ," ,loss ",loss)