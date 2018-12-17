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
from __future__ import print_function
import argparse
import tensorflow as tf
import sys
from tensorflow.python import debug as tf_debug
import os
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Create model
def simple_mlp(x):
    hidden_layer = tf.layers.dense(inputs=x, units=512)
    out_layer = tf.layers.dense(inputs=hidden_layer, units=10)
    return out_layer


# Parameters
learning_rate = 0.01
num_steps = 640
batch_size = 64
display_step = 100

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

tf.get_default_graph().seed = 1

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Construct model
model_logits = simple_mlp(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(model_logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
log_dir = "/tmp/tf_sim/mnist"

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', default='ngraph',
                    help='device [gpu, cpu, ngraph]')
args = parser.parse_args()

def run_session():
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # Run the initializer
        print(sess.run(init))

        # dumps_dir = 'file:///tmp/tf_dump/mnist_mlp'
        # run_options = tf.RunOptions()
        # tf_debug.watch_graph(
        #     run_options,
        #     sess.graph,
        #     debug_urls=[dumps_dir]
        # )

        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=False)
            # Run optimization op (backprop)
            res = sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})#, options=run_options)

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        writer.close()

if args.device == 'ngraph':
    import ngraph_bridge
    run_session()
elif args.device == 'gpu':
    with tf.device('/gpu:0'):
        run_session()
else:
    with tf.device('/cpu'):
        run_session()

print("Tensorboard log saved to: %s" % log_dir)
