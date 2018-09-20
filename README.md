# Bridge TensorFlow* to run on Intel® nGraph™ backends

This directory contains the code needed to build a TensorFlow 
plugin to the Intel® nGraph™ Compiler. As part of a TensorFlow 
toolchain, nGraph can speed up training workloads using CPU; or 
the nGraph Library and runtime suite can be used to customize 
and deploy Deep Learning inference models that will "just work" 
with variety of nGraph-enabled backends: CPU, GPU, and custom 
silicon.  

*   [Linux instructions](#linux-instructions)
*   [OS X instructions](#os-x-instructions)
*   [Debugging](#debugging)
*   [Support](#support)
*   [About nGraph](#about-ngraph)



## Linux instructions

You can connect nGraph to an existing TensorFlow installation if 
you're running TensorFlow v1.11.0-rc1 or greater. For versions 
released previous to that, we recommend starting with a clean 
environment and re-building with the instructions in Option 2.

### Option 1: Use an existing TensorFlow installation

1. If you already have a requisite version resultant from following 
   the [linux-based install instructions on the TensorFlow website], 
   you must also instantiate a specific kind of `virtualenv`  to 
   be able to proceed with the `ngraph-tf` bridge installation. For 
   systems with Python 3.n or Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
    
2. Checkout `v0.5.0` from the `ngraph-tf` repo and build the bridge
   as follows: 
   
        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf
        git checkout v0.5.0
        mkdir build
        cd build
        cmake ..
        make -j <number_of_processor_cores_on_system>
        make install 
        pip install -U python/dist/ngraph-0.5.0-py2.py3-none-linux_x86_64.whl


### Option 2: Use the Build nGraph bridge from source using TensorFlow source

To run unit tests, or if you are planning to contribute, install the nGraph 
bridge using the TensorFlow source tree. 

#### Prepare the build environment

The installation prerequisites are the same as described in the TensorFlow 
[prepare environment] for linux.

1. We use the standard build process which is a system called "bazel". These 
   instructions were tested with [bazel version 0.16.0]. 

        wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh      
        chmod +x bazel-0.16.0-installer-linux-x86_64.sh
        ./bazel-0.16.0-installer-linux-x86_64.sh --user

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

3. Ensure that all the TensorFlow dependencies are installed, as per the
   TensorFlow [prepare environment] for linux. :exclamation: You do not 
   need CUDA in order to use the ngraph-tf bridge.

4. Additional dependencies.
   - Install ```apt-get install libicu-dev``` to avoid the following (potential) error:
     ```unicode/ucnv.h: No such file or directory```.


#### Installation

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine. 

     :warning: You need the following version of TensorFlow: `v1.10.0`

        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v1.10.0
        git status
        HEAD detached at v1.10.0
   
2. You must instantiate a specific kind of `virtualenv`  to be able to proceed 
   with the `ngraph-tf` bridge installation. For systems with Python 3.n or 
   Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
   
3. Now run `./configure` and choose `no` for all the questions when prompted to build TensorFlow.

    Note that if you are running TensorFlow on a Skylake family processor then select
    `-march=broadwell` when prompted to specify the optimization flags:
    ```
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: -march=broadwell
    ```
    This is due to an issue in TensorFlow which is being actively worked on: 
    https://github.com/tensorflow/tensorflow/issues/17273

4. Prepare the pip package and the TensorFlow C++ library:

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
        bazel-bin/tensorflow/tools/pip_package/build_pip_package ./

     :exclamation: You may run into the following error:
    ```
    AttributeError: 'int' object attribute '__doc__' is read-only
    Target //tensorflow/tools/pip_package:build_pip_package failed to build
    Use --verbose_failures to see the command lines of failed build steps.
    ```
    in which case you need to install enum34:
    ```
    pip install enum34
    ```

    You may also need to install a Python package named ```mock``` to prevent an import 
    error during the build process.

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

        pip install -U ./tensorflow-1.*whl
   
6. Now clone the `ngraph-tf` repo one level above -- in the 
  *parent* directory of the `tensorflow` repo cloned in step 1:

        cd ..
        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf

7. Next, build and install nGraph bridge. 
   :warning: Run the ngraph-tf build from within the `virtualenv`.

        mkdir build
        cd build
        cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR=<location of the TensorFlow source directory> ..
        make -j <your_processor_cores>
        make install 
        pip install -U python/dist/<ngraph-0.5.0-py2.py3-none-linux_x86_64.whl>

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies. The resulting plugin [DSO] is named `libngraph_bridge.so`.

Once the build and installation steps are complete, you can start using TensorFlow 
with nGraph backends. 

Note: The actual filename for the pip package may be different as it's version 
dependent. Please check the `build/python/dist` directory for the actual pip wheel.

#### Running tests

To run the C++ unit tests, please do the following:

1. Go to the build directory and run the following command:
    ```
    cd test
    ./gtest_ngtf
    ```
Next is to run a few DL models to validate the end-to-end functionality.

2. Go to the ngraph-tf/examples directory and run the following models.
    ```
    cd examples/mnist
    python mnist_fprop_only.py \
        --data_dir <input_data_location> 
    ```

## OS X instructions

The build and installation instructions are idential for Ubuntu 16.04 and OS X.

### Running tests

1. Add `<path-to-tensorflow-repo>/bazel-out/darwin-py3-opt/bin/tensorflow` and `<path-to-ngraph-tf-repo>/build/ngraph/ngraph_dist/lib` to your `DYLD_LIBRARY_PATH`
2. Follow the C++ and Python instructions from the Linux based testing described above.

## Debugging

See the instructions provided in the [diagnostics] directory.

## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea for how to 
improve it:

* Share your proposal via [GitHub issues].
* Make sure your patch is in line with Google style by setting up your git 
  `pre-commit` hooks. First, ensure `clang-format` is in your path, then:
   
        pip install pre-commit autopep8 pylint
        pre-commit install
   
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.


## About Intel® nGraph™

See the full documentation here:  <http://ngraph.nervanasys.com/docs/latest>


## Future plans

[linux-based install instructions on the TensorFlow website]:https://www.tensorflow.org/install/install_linux
[tensorflow]:https://github.com/tensorflow/tensorflow.git
[open-source C++ library, compiler and runtime]: http://ngraph.nervanasys.com/docs/latest/
[DSO]:http://csweb.cs.wfu.edu/~torgerse/Kokua/More_SGI/007-2360-010/sgi_html/ch03.html
[Github issues]: https://github.com/NervanaSystems/ngraph-tf/issues
[pull request]: https://github.com/NervanaSystems/ngraph-tf/pulls
[bazel version 0.16.0]: https://github.com/bazelbuild/bazel/releases/tag/0.16.0
[prepare environment]: https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux
[diagnostics]:diagnostics/README.md

 
 
