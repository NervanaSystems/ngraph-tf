# TensorFlow C++ application example using nGraph

This directory contains an example C++ application that uses TensorFlow and nGraph. The example creates a simple computation graph and executes using nGraph computation backend.

The application is linked with TensorFlow C++ library and nGraph-TensorFlow bridge library. 

## prerequisites

The example application requires the following include files

1. nGraph core header files
2. nGraph-TensorFlow bridge header files
3. TensorFlow header files

The application links with the following dynamic shared object (DSO) libraries

1. libngraph_bridge.so
2. libtensorflow_framework.so
3. libtensorflow_cc.so

Build the ngraph-tf by executing `build_gtf.py` as per the [Option1] instructions in the main readme. All the files needed to build this example application is located in the ngraph-tf/build directory.

## Build the example

Create a working directory at the parent directory of `ngraph-tf` and copy the `Makefile` and the `hello_tf.cpp`. Next create a directory called `lib` in this working directory and copy the following DSO files:

1. ../ngraph-tf/build/artifacts/

[Option1]: ../../README.md#option-1-use-a-pre-built-ngraph-tensorflow-bridge