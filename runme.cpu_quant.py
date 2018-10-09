import tensorflow as tf, numpy as np
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
import pdb, os
import gzip, shutil
from mnist import MNIST
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin
import ngraph

# https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
def download_file(fname, target_dir, force=False):
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
  
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin('http://yann.lecun.com/exdb/mnist/', fname)
        urlretrieve(url, target_fname)

    unzip_name = target_fname[:-3]
    with gzip.open(target_fname, 'rb') as f_in:
      with open(unzip_name, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

    return unzip_name

def get_whole_dataset(mnist_dir):
  dl_fl_names = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
  print ([download_file(item, mnist_dir) for item in dl_fl_names])



graphdef = graph_pb2.GraphDef()
graph_file="/nfs/site/home/jbobba/final_int8_mnist.pbtxt"
f = open(graph_file, "r")
text_format.Merge(f.read(), graphdef)

for node in graphdef.node:
  print(node.name)
  print("  " + node.op)
  print("  inputs:")
  for input in node.input:
    print("    " + input)
#pdb.set_trace()

mnist_dir = './mnist'
get_whole_dataset(mnist_dir)  # create mnist_dir and download data in here

mndata = MNIST(mnist_dir)
images, labels = mndata.load_testing()


with tf.Session() as sess:
  graph = tf.import_graph_def(graphdef)
  #placeholders = [ op for op in tf.get_default_graph().get_operations() if op.type == "Placeholder"]
  intensor1 = tf.get_default_graph().get_tensor_by_name('import/Placeholder:0')
  intensor2 = tf.get_default_graph().get_tensor_by_name('import/Placeholder_1:0')
  #print placeholders
  #pool1/MaxPool_eightbit_quantized  QMP
  #pool1/MaxPool_eightbit_quantize_conv1/Relu  Qv2
  #import/accuracy
  #conv1/Relu
  outtensor = tf.get_default_graph().get_tensor_by_name('import/pool1/MaxPool_eightbit_quantize_conv1/Relu:0')
  #print [ op for op in tf.get_default_graph().get_operations() if op.type == "QuantizeV2"]

  outvals = sess.run([outtensor], feed_dict = {intensor1 : np.array([images[0]]), intensor2 : np.array([labels[0]])})

print ('===============')
ngraph.disable()
with tf.Session() as sess:
  graph = tf.import_graph_def(graphdef)
  intensor1 = tf.get_default_graph().get_tensor_by_name('import/Placeholder:0')
  intensor2 = tf.get_default_graph().get_tensor_by_name('import/Placeholder_1:0')
  outtensor = tf.get_default_graph().get_tensor_by_name('import/pool1/MaxPool_eightbit_quantize_conv1/Relu:0')
  outvals_tf = sess.run([outtensor], feed_dict = {intensor1 : np.array([images[0]]), intensor2 : np.array([labels[0]])})

for t1, t2 in zip(outvals, outvals_tf):
  print(np.linalg.norm(t1 - t2))
print('hello')
