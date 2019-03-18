import tensorflow as tf
import ngraph_bridge
import os

os.putenv("NGRAPH_TG_VLOG_LEVEL", "5")
os.putenv("NGRAPH_TF_LOG_PLACEMENT", "1")
os.putenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1")

sess = tf.Session()
a = tf.convert_to_tensor([[40, 30, 20, 10], [10, 20, 15, 70]])
b = tf.nn.top_k(a, 3, False)

print(sess.run(b))
