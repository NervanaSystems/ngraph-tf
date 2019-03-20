#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

rm -f .bazelrc
if python -c "import tensorflow" &> /dev/null; then
    echo 'using installed tensorflow'
else
    pip install tensorflow==1.12.0
    pip install tensorflow_estimator
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CXX_ABI=( $(python -c 'import tensorflow as tf; print(" ".join(str(tf.__cxx11_abi_flag__)))') )


write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}

# Write the CXX ABI to the file 
echo "CXX_ABI = ['-D_GLIBCXX_USE_CXX11_ABI=$TF_CXX_ABI']" > cxx_abi_option.bzl

# Create symbolic links to WORKSPACE and BUILD so that this directory
# is now ready to run bazel based builds
ln -sf bazel/BUILD .
ln -sf bazel/WORKSPACE .
ln -sf bazel/tf_configure .

