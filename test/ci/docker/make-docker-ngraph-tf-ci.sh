#!  /bin/sh

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

# Script parameters:
#
# $1 ImageID    Required: ID of the ngtf_bridge_ci docker image to use
# $2 ImageType  Optional: Type of docker image to build
#
# Additional parameters are passed to docker-build command
#
# Within .intel.com, appropriate proxies are applied
#
# Script environment variable parameters:
#
# NG_TF_PY_VERSION   Optional: Set python major version ("2" or "3", default=2)

set -e  # Fail on any command with non-zero exit

IMAGE_ID="$1"
if [ -z "${IMAGE_ID}" ] ; then  # Parameter 1 is REQUIRED
    echo 'Please provide an image version as the only argument'
    exit 1
else
    shift 1  # We found parameter one, remove it from $@
fi

IMAGE_TYPE="$1"  # Second parameter has been shifted to be first parameter
if [ -z "${IMAGE_TYPE}" ] ; then  # PARAMETER 2 is OPTIONAL
    IMAGE_TYPE='default'
else
    shift 1  # We found parameter two, remove it from $@
fi

# Set defaults

echo "DBG: IMAGE_TYPE before case is ${IMAGE_TYPE}"
case "${IMAGE_TYPE}" in
    default)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    default_py27)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    default_py35)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py3'
        IMAGE_NAME='ngraph_tf_ci_py3'
        ;;
    ubuntu1604_gcc48_py27)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci.ubuntu1604-gcc48-py27'
        IMAGE_NAME='ngraph_tf_ci_ubuntu1604_gcc_48_py27'
        ;;
    ubuntu1604_gcc48_py35)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci.ubuntu1604-gcc48-py35'
        IMAGE_NAME='ngraph_tf_ci_ubuntu1604_gcc48_py35'
        ;;
    *)
        echo "INTERNAL ERROR: unrecognized IMAGE_TYPE (${IMAGE_TYPE})"
        exit 1
        ;;
esac


# The NG_TF_PY_VERSION takes precedence over the optional IMAGE_TYPE parameter,
# because NG_TF_PY_VERSION existed first and we need to maintain backward
# compatibility (for now)
case "${NG_TF_PY_VERSION}" in
    2)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py2'
        IMAGE_NAME='ngraph_tf_ci_py2'
        ;;
    3)
        DOCKER_FILE='Dockerfile.ngraph-tf-ci-py3'
        IMAGE_NAME='ngraph_tf_ci_py3'
        ;;
    *)
        # Do nothing if NG_TF_PY_VERSION is explicitly not set
        ;;
esac

set -u  # No unset variables after this point

# Show in log what is being build
echo "make-docker-ngraph-tf-ci is building the following:"
echo "    IMAGE_TYPE: ${IMAGE_TYPE}"
echo "    DOCKER_FILE: ${DOCKER_FILE}"
echo "    IMAGE_NAME: ${IMAGE_NAME}"
echo "    IMAGE_ID: ${IMAGE_ID}"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

# Context is the maint-jenkins directory, to avoid including all of
# ngraph-tensorflow-1.3 in the context.
#
# The $@ allows us to pass command-line options easily to docker build.
# Note that a "shift" is done above to remove the IMAGE_ID from the cmd line.
#
dbuild_cmd="docker build  --rm=true \
            ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
            $@ \
            -f=${DOCKER_FILE}  -t=${IMAGE_NAME}:${IMAGE_ID}  ."
echo "Docker build command: ${dbuild_cmd}"
$dbuild_cmd
