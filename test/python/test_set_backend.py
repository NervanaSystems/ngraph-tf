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
"""nGraph TensorFlow bridge cast operation test

"""
from __future__ import print_function 
import pytest

import tensorflow as tf

import ngraph_config


backend_cpu = "CPU"
backend_interpreter = "INTERPRETER"
print ("Check")
ngraph_config.enable()

print("Try Is Supported")

if ngraph_config.is_supported_backend(backend_cpu):
    print("Backend" + backend_cpu + " is supported")
    print("Try Set Backend")
    ngraph_config.set_backend(backend_cpu)

if ngraph_config.is_supported_backend(backend_interpreter):
    print("Backend" + backend_interpreter + " is supported")
    print("Try Set Backend")
    ngraph_config.set_backend(backend_interpreter)
