#!  /bin/bash

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
#
# Script environment variable parameters:
#
# NG_TF_KW_SERVER      Required: KW server to use
# NG_TF_KW_LTOKEN_DIR  Required: ltoken file containing KW authorization
# NG_TF_KW_TOOLS       Required: directory of Klocwork tools to use
# NG_TF_TF_VERSION     Optional: which TF version to download
# NG_TF_PY_VERSION     Optional: Set python major version ("2" or "3", default=2)

set -e  # Fail on any command with non-zero exit

#DISABLED tf_dir="${2}"
#DISABLED if [ ! -d "${tf_dir}" ] ; then
#DISABLED     echo 'Please provide the name of the tensorflow directory you want to build, as the 2nd parameter'
#DISABLED    exit 1
#DISABLED fi

# Set defaults

if [ -z "${NG_TF_KW_SERVER}" ] ; then
    echo 'NG_TF_KW_SERVER must be set to the URL of the KW server for this scan'
    exit 1
fi

if [ -z "${NG_TF_KW_LTOKEN_DIR}" ] ; then
    echo 'NG_TF_KW_LTOKEN_DIR must be set to the path of the Klocwork ltoken authorization file'
    exit 1
fi

if [ -z "${NG_TF_KW_TOOLS}" ] ; then
    echo 'NG_TF_KW_TOOLS must be set to the name of the Klocwork tools directory you want to use'
    exit 1
fi

if [ -z "${NG_TF_TF_VERSION}" ] ; then
    NG_TF_TF_VERSION=''  # Use latest TF wheel from the internet
fi

if [ -z "${NG_TF_PY_VERSION}" ] ; then
    NG_TF_PY_VERSION='3'  # Default is Python 3
fi

# Note that the docker image must have been previously built using the
# make-docker-ngraph-tf-ci.sh script (in the same directory as this script).
#
case "${NG_TF_PY_VERSION}" in
    2)
        echo 'Only Python version 3 is currently supported for Klocwork builds and scans'
        exit 1
        # DISABLED  IMAGE_CLASS='ngraph_tf_ci_klocwork_py2'
        ;;
    3)
        IMAGE_CLASS='ngraph_tf_klocwork_ubuntu1604_py35'
        ;;
    *)
        echo 'NG_TF_PY_VERSION must be set to "2", "3", or left unset (default is "2")'
        exit 1
        ;;
esac
IMAGE_ID="${1}"
if [ -z "${IMAGE_ID}" ] ; then
    echo 'Please provide an image version as the first parameter'
    exit 1
fi

# Check if we have a docker ID of image:ID, or just ID
# Use || true to make sure the exit code is always zero, so that the script is
# not killed if ':' is not found
long_ID=`echo ${IMAGE_ID} | grep ':' || true`

# If we have just ID, then IMAGE_CLASS AND IMAGE_ID have
# already been set above
#
# Handle case where we have image:ID
if [ ! -z "${long_ID}" ] ; then
    IMAGE_CLASS=` echo ${IMAGE_ID} | sed -e 's/:[^:]*$//' `
    IMAGE_ID=` echo ${IMAGE_ID} | sed -e 's/^[^:]*://' `
    # TODO: set python version here based on presence of _py3
fi

# Find the top-level bridge directory, so we can mount it into the docker
# container
bridge_dir="$(realpath ../../..)"

bridge_mountpoint='/home/dockuser/ngraph-tf'
kwtools_mountpoint='/home/dockuser/kwtools'
#DISABLED tf_mountpoint='/home/dockuser/tensorflow'

# Set up a bunch of volume mounts
volume_mounts='-v /dataset:/dataset'
volume_mounts="${volume_mounts} -v ${bridge_dir}:${bridge_mountpoint}"
volume_mounts="${volume_mounts} -v ${kwtools_dir}:${kwtools_mountpoint}"
volume_mounts="${volume_mounts} -v ${NG_TF_KW_TOOLS}:/home/dockuser/kwtools"
volume_mounts="${volume_mounts} -v ${NG_TF_KW_LTOKEN_DIR}:/home/dockuser/ltoken-dir"

# Set up optional environment variables
env_vars=''
if [ ! -z "${NG_TF_KW_SERVER}" ] ; then
  env_vars="${env_vars} --env NG_TF_KW_SERVER=${NG_TF_KW_SERVER}"
fi
if [ ! -z "${NG_TF_KW_LTOKEN_DIR}" ] ; then
  env_vars="${env_vars} --env NG_TF_KW_LTOKEN_DIR=${NG_TF_KW_LTOKEN_DIR}"
fi
if [ ! -z "${NG_TF_KW_TOOLS}" ] ; then
  env_vars="${env_vars} --env NG_TF_KW_TOOLS=${NG_TF_KW_TOOLS}"
fi
if [ ! -z "${NG_TF_TF_VERSION}" ] ; then
  env_vars="${env_vars} --env NG_TF_TF_VERSION=${NG_TF_TF_VERSION}"
fi
if [ ! -z "${NG_TF_PY_VERSION}" ] ; then
  env_vars="${env_vars} --env NG_TF_PY_VERSION=${NG_TF_PY_VERSION}"
fi

set -u  # No unset variables after this point

RUNASUSER_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-as-user.sh"
BUILD_KW_SCRIPT="${bridge_mountpoint}/test/ci/docker/docker-scripts/run-ngraph-tf-klocwork.sh"
# FOR DEBUGGING: BUILD_KW_SCRIPT="/bin/bash"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--env http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--env https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

set -x
docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD="${BUILD_KW_SCRIPT}" \
       ${env_vars} \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
       ${volume_mounts} \
       "${IMAGE_CLASS}:${IMAGE_ID}" "${RUNASUSER_SCRIPT}"
