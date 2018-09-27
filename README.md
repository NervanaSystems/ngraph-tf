# nGraph-Tensorflow : Intel® nGraph™ Compiler and Runtime for TensorFlow

nGraph-Tensorflow Bridge enables TensorFlow to run with [Intel® nGraph™](https://github.com/NervanaSystems/ngraph), Compiler and 
Runtime engine, speeding up training and inference workloads on hardware supported by nGraph: CPU, GPU, and custom silicon like the [Intel® Nervana™ NNP](https://itpeernetwork.intel.com/inteldcisummit-artificial-intelligence/). It integrates seamlessly with Tensorflow, allowing developers the flexibility to switch amongst different hardwares, by making minimum to no change to their code.

*   [Build with Linux](#linux-instructions)
*   [Build using OS X](#using-os-x)
*   [Debugging](#debugging)
*   [Support](#support)
*   [How to Contribute](#how-to-contribute)


## Linux instructions
There are 3 ways to install nGraph enabled for tensorflow.

| Option          | Build TF    | Build nGraph |    
|                 | from Source | from Source  |            
|:---------------:|:-----------:|:------------:|:---------: 
| [1](#Option-1:) |   No        |  Yes         | 
|  :--:           |  :--:       | :--:         | :----:      
| [2](#Option-2:) |  Yes        |  Yes         | Builds unit tests. Recommended 
|                 |             |              | if planning to contribute to 
|                 |             |              | nGraph-Tensorflow
|  :--:           |  :--:       | :--:         | :----:      
| [3](#Option-3:) |  Yes        |  No          | nGraph enabled by default             


### Option 1: Use an existing TensorFlow v1.11.0-rc2 (or greater) installation

1. You need to instantiate a specific kind of `virtualenv`  to 
   be able to proceed with the `ngraph-tf` bridge installation. For 
   systems with Python 3.n or Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
    
2. Install TensorFlow v1.11.0-rc2. Note that this is a pre-release so you need 
   to use the following steps to install this:

        pip install tensorflow==1.11.0rc2
   Note: When TensorFlow release version `v1.11.0` is available, 
   update your version of TensorFlow using the following command:

        pip install -U tensorflow

3. Checkout `v0.6.0` from the `ngraph-tf` repo and build the bridge
   as follows: 
   
        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf
        git checkout v0.6.0
        mkdir build
        cd build
        cmake ..
        make -j <number_of_processor_cores_on_system>
        make install 
        pip install -U python/dist/ngraph-0.6.0-py2.py3-none-linux_x86_64.whl

To enable nGraph in your python scripts
        import ngraph
    
### Option 2: Build nGraph using TensorFlow source

To run unit tests, or if you are planning to contribute, install the nGraph 
bridge using the TensorFlow source tree as follows: 

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

#### Installation

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine. 

     :warning: You need the following version of TensorFlow: `v1.11.0-rc2`

        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v1.11.0-rc2
        git status
        HEAD detached at v1.11.0-rc2
   
2. You must instantiate a specific kind of `virtualenv`  to be able to proceed 
   with the `ngraph-tf` bridge installation. For systems with Python 3.n or 
   Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
        
   Note: Depending of specific version of the Python and components already
   installed on your system - the list of dependent Python components vary. 
   Typically the following components are needed: `numpy mock keras keras_application`.
   Install them if your Python environment doesn't have them already. 
   
3. Now run `./configure` and choose `no` for the following when prompted to build TensorFlow.

    CUDA support:
    
        Do you wish to build TensorFlow with CUDA support? [y/N]: N
        No CUDA support will be enabled for TensorFlow.
    
    nGraph support:

        Do you wish to build TensorFlow with nGraph support? [y/N]: N
        No nGraph support will be enabled for TensorFlow.

    Since you are building nGraph using an existing TensorFlow build, you cannot respond with `y`
    for the above step. This will result in conflicts as there will be two versions of
    nGraph - one embedded within TensorFlow and the other you build and loaded. 
    
    If you want to use the nGraph embedded within TensorFlow, see the 
    [following section](#option-3).

    Note that if you are running TensorFlow on a Skylake family processor then select
    `-march=broadwell` when prompted to specify the optimization flags:
    
        Please specify optimization flags to use during compilation 
        when bazel option "--config=opt" is specified 
        [Default is -march=native]: -march=broadwell
    
    This is due to an issue in TensorFlow which is being actively worked on: 
    https://github.com/tensorflow/tensorflow/issues/17273

4. Prepare the pip package and the TensorFlow C++ library:

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
        bazel-bin/tensorflow/tools/pip_package/build_pip_package ./

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

        pip install -U ./tensorflow-1.*whl
   
6. Now clone the `ngraph-tf` repo one level above -- in the 
  *parent* directory of the `tensorflow` repo cloned in step 1:

        cd ..
        git clone https://github.com/NervanaSystems/ngraph-tf.git
        cd ngraph-tf
        git checkout v0.6.0


7. Next, build and install nGraph bridge. 
   :warning: Run the ngraph-tf build from within the `virtualenv`.

        mkdir build
        cd build
        cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR=<path to TensorFlow source directory> ..
        make -j <your_processor_cores>
        make install 
        pip install -U python/dist/<ngraph-0.6.0-py2.py3-none-linux_x86_64.whl>

This final step automatically downloads the necessary version of `ngraph` and 
the dependencies.

Once the build and installation steps are complete, you can start using TensorFlow 
with nGraph backends. 

To enable nGraph in your python scripts
        import ngraph

Note: The actual filename for the pip package may be different as it's version 
dependent. Please check the `build/python/dist` directory for the actual pip wheel.

### Option 3: Build nGraph with TensorFlow Source

nGraph is being added to the TensorFlow source tree. When built with this option, there is **no need to separately build `ngraph-tf` or use `pip` to install the ngraph module**. With this configuration, your TensorFlow model scripts will work with nGraph without any changes. 

#### Prepare the build environment

The installation prerequisites are the same as described in the TensorFlow [prepare environment] for linux.

1. We use the standard build process which is a system called "bazel". These 
   instructions were tested with [bazel version 0.16.0]. 

        wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh      
        chmod +x bazel-0.16.0-installer-linux-x86_64.sh
        ./bazel-0.16.0-installer-linux-x86_64.sh --user

2. Add and source the ``bin`` path to your ``~/.bashrc`` file in order to be 
   able to call bazel from the user's installation we set up:

        export PATH=$PATH:~/bin
        source ~/.bashrc   

#### Installation

1. Once TensorFlow's dependencies are installed, clone the source of the 
   [tensorflow] repo to your machine. 

     :warning: You need the following version of TensorFlow: `v1.11.0-rc2`

        git clone https://github.com/tensorflow/tensorflow.git
        cd tensorflow
        git checkout v1.11.0-rc2
        git status
        HEAD detached at v1.11.0-rc2
   
2. You must instantiate a specific kind of `virtualenv`  to be able to proceed 
   with the `ngraph-tf` bridge installation. For systems with Python 3.n or 
   Python 2.7, these commands are

        virtualenv --system-site-packages -p python3 your_virtualenv 
        virtualenv --system-site-packages -p /usr/bin/python2 your_virtualenv  
        source your_virtualenv/bin/activate # bash, sh, ksh, or zsh
        
   Note: Depending of specific version of the Python and components already
   installed on your system - the list of dependent Python components vary. 
   Typically the following components are needed: `numpy mock keras keras_application`.
   Install them if your Python environment doesn't have them already. 
   
3. Now run `./configure` and choose the following when prompted

    CUDA support:
    
        Do you wish to build TensorFlow with CUDA support? [y/N]: N
        No CUDA support will be enabled for TensorFlow.
    
    nGraph support:

        Do you wish to build TensorFlow with nGraph support? [y/N]: y
        No nGraph support will be enabled for TensorFlow.

    Note that if you are running TensorFlow on a Skylake family processor then select
    `-march=broadwell` when prompted to specify the optimization flags:
    
        Please specify optimization flags to use during compilation 
        when bazel option "--config=opt" is specified 
        [Default is -march=native]: -march=broadwell
    
    This is due to an issue in TensorFlow which is being actively worked on: 
    https://github.com/tensorflow/tensorflow/issues/17273

4. Prepare the pip package and the TensorFlow C++ library:

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
        bazel-bin/tensorflow/tools/pip_package/build_pip_package ./

5. Install the pip package, replacing the `tensorflow-1.*` with your 
   version of TensorFlow:

        pip install -U ./tensorflow-1.*whl

Note: The version that is available with TensorFlow usually lags the features and bug fixes available in the `master` branch of this repository.

### Running tests

To run the C++ unit tests,

* Go to the build directory and run the following commands:

    cd test
    ./gtest_ngtf

You can run a few of your own DL models to validate the end-to-end 
functionality. Also, you can use the `ngraph-tf/examples` directory and try to 
run the following model with some MNIST data on your local machine: 

        cd examples/mnist
        python mnist_fprop_only.py --data_dir <input_data_location> 

## Using OS X 

The build and installation instructions are idential for Ubuntu 16.04 and OS X. 

### Running tests

Export the appropriate paths to your build location; OS X uses the `DYLD_` prefix:

    export DYLD_LIBRARY_PATH=/bazel-out/darwin-py3-opt/bin/tensorflow:$DYLD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=/build/ngraph/ngraph_dist/lib:$DYLD_LIBRARY_PATH

Then follow ["Running tests"](#Running-tests) on Linux as described above. 

## Debugging

See the instructions provided in the [diagnostics] directory.

https://github.com/NervanaSystems/ngraph-tf/blob/master/diagnostics/README.md


## Support

Please submit your questions, feature requests and bug reports via [GitHub issues].

## How to Contribute

We welcome community contributions to nGraph. If you have an idea for how to 
improve it:

* Share your proposal via [GitHub issues].
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
[ops]:http://ngraph.nervanasys.com/docs/latest/ops/index.html
[nGraph]:https://github.com/NervanaSystems/ngraph 
[ngraph-tf bridge]:https://github.com/NervanaSystems/ngraph-tf 
 
