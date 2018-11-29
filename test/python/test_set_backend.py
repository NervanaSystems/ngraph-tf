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
import pytest
import tensorflow as tf
import ngraph_bridge


# Test ngraph_bridge config options
def test_set_backend():
    print("Number of supported backends ", ngraph_bridge.backends_len())
    supported_backends = ngraph_bridge.list_backends()
    print(" ****** Supported Backends ****** ")
    for backend_name in supported_backends:
        print (backend_name)
    print(" ******************************** ")

    backend_cpu = 'CPU'
    backend_interpreter = 'INTERPRETER'

    if ngraph_bridge.is_supported_backend(backend_cpu):
        ngraph_bridge.set_backend(backend_cpu)
        currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
        assert currently_set_backend == backend_cpu

    if ngraph_bridge.is_supported_backend(backend_interpreter):
        ngraph_bridge.set_backend(backend_interpreter)
        currently_set_backend = ngraph_bridge.get_currently_set_backend_name()
        assert currently_set_backend == backend_interpreter
