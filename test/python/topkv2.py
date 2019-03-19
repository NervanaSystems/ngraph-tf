import tensorflow as tf
import ngraph_bridge
import os

os.putenv("NGRAPH_TG_VLOG_LEVEL", "5")
os.putenv("NGRAPH_TF_LOG_PLACEMENT", "1")
os.putenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1")

sess = tf.Session()
a = tf.convert_to_tensor([[40.0, 30.0, 20.0, 10.0], [10.0, 20.0, 15.0, 70.0]])
b = tf.nn.top_k(a, 3)

print(sess.run(b))
