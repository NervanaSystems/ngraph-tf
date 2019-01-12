# TensorFlow C++ application example using nGraph

This directory contains an example C++ application that uses TensorFlow and nGraph. The example creates a simple computation graph and executes using nGraph computation backend.

The application is linked with TensorFlow C++ library and nGraph-TensorFlow bridge library. 

## prerequisites

The example application requires nGraph-TensorFlow bridge to be built first. Build the ngraph-tf by executing `build_gtf.py` as per the [Option 1] instructions in the main readme. All the files needed to build this example application is located in the ngraph-tf/build directory.

### Dependencies

The following include files are needed to build the C++ application:

1. nGraph core header files
2. nGraph-TensorFlow bridge header files
3. TensorFlow header files

The application links with the following dynamic shared object (DSO) libraries

1. libngraph_bridge.so
2. libtensorflow_framework.so
3. libtensorflow_cc.so


## Build the example

### Update the Makefile

If you are using clang or gcc 4.8 to build ngraph-tf then change the `CXX_ABI_FLAG` value to 0. Also, change the `NGRAPH_TF_DIR` pointing to the location of the `ngraph-tf` on your system as needed.

### Build

Run the make command to build the application that will produce the executable: `hello_tf`.

### Run

Before running the application, set the `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` for MacOS) to point to the `lib` directory created earlier where all the DSO files are previously copied.

Next run the executable `./hello_tf`

[Option 1]: ../../README.md#option-1-use-a-pre-built-ngraph-tensorflow-bridge
