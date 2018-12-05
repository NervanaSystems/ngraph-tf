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

# This script is designed to be called from within a docker container.
# It is installed into a docker image.  It will not run outside the container.

set -e  # Make sure we exit on any command that returns non-zero
set -o pipefail # Make sure cmds in pipe that are non-zero also fail immediately


# Make sure NG_TF_KW_SERVER is set
if [ -z "${NG_TF_KW_SERVER}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "NG_TF_KW_SERVER must be set when running Klocwork automation" )
    exit 1
fi

# Make sure NG_TF_TF_VERSION is set, for set -u below
if [ -z "${NG_TF_TF_VERSION}" ] ; then
    NG_TF_TF_VERSION=''
fi

# Default is Python 2, but can override with NG_TF_PY_VERSION env. variable
export PYTHON_VERSION_NUMBER="${NG_TF_PY_VERSION}"
if [ -z "${PYTHON_VERSION_NUMBER}" ] ; then
    PYTHON_VERSION_NUMBER=2
fi
export PYTHON_BIN_PATH="/usr/bin/python${PYTHON_VERSION_NUMBER}"
export PYTHON_PIP_CMD="pip${PYTHON_VERSION_NUMBER}"

set -u  # No unset variables after this point

# Set up some important known directories
bridge_dir='/home/dockuser/ngraph-tf'                      # DO NOT CHANGE -- KW issues reported via this path
bbuild_dir='/home/dockuser/BUILD-KW'                       # DO NOT CHANGE -- KW issues reported via this path
venv_dir="/home/dockuser/venv_py${PYTHON_VERSION_NUMBER}"  # DO NOT CHANGE -- KW issues reported via this path
ci_dir="${bridge_dir}/test/ci/docker"
ngraph_wheel_dir="${bbuild_dir}/python/dist"

# Set up paths to the Klocwork tools
kw_project='ngraph-tf'
kw_tools='/home/dockuser/kwtools'
kw_ltoken='/home/dockuser/ltoken-dir/ltoken'
kw_bin="${kw_tools}/bin"

# HOME is expected to be /home/dockuser.  See script run-as-user.sh, which
# sets this up.

echo "In $(basename ${0}):"
echo ''
echo "  bridge_dir=${bridge_dir}"
echo "  bbuild_dir=${bbuild_dir}"
echo "  ci_dir=${ci_dir}"
echo "  venv_dir=${venv_dir}"
echo "  ngraph_wheel_dir=${ngraph_wheel_dir}"
echo ''
echo "  kw_project=${kw_project}"
echo "  kw_tools=${kw_tools}"
echo "  kw_ltoken=${kw_ltoken}"
echo "  kw_bin=${kw_bin}"
echo ''
echo "  HOME=${HOME}"
echo "  PYTHON_VERSION_NUMBER=${PYTHON_VERSION_NUMBER}"
echo "  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
echo "  PYTHON_PIP_CMD=${PYTHON_PIP_CMD}"

# Do some up-front checks, to make sure necessary directories are in-place and
# build directories are not-in-place

if [ -d "${bbuild_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Bridge build directory already exists -- please remove it before calling this script: ${bbuild_dir}" )
    exit 1
fi

if [ -d "${venv_dir}" ] ; then
    ( >&2 echo '***** Error: *****' )
    ( >&2 echo "Virtual-env build directory already exists -- please remove it before calling this script: ${venv_dir}" )
    exit 1
fi

# Make sure the Bazel cache is in /tmp, as docker images have too little space
# in the root filesystem, where /home (and $HOME/.cache) is.  Even though we
# may not be using the Bazel cache in the builds (in docker), we do this anyway
# in case we decide to turn the Bazel cache back on.
echo "Adjusting bazel cache to be located in /tmp/bazel-cache"
rm -fr "$HOME/.cache"
mkdir /tmp/bazel-cache
ln -s /tmp/bazel-cache "$HOME/.cache"


xtime="$(date)"
echo  ' '
echo  "===== Setting Up Virtual Environment for Tensorflow Wheel at ${xtime} ====="
echo  ' '

# Make sure the bash shell prompt variables are set, as virtualenv crashes
# if PS2 is not set.
PS1='prompt> '
PS2='prompt-more> '
virtualenv --system-site-packages -p "${PYTHON_BIN_PATH}" "${venv_dir}"
source "${venv_dir}/bin/activate"


xtime="$(date)"
echo  ' '
echo  "===== Installing the TensorFlow wheel from the Internet at ${xtime} ====="
echo  ' '

if [ -z "${NG_TF_TF_VERSION}" ] ; then
    ${PYTHON_PIP_CMD} install tensorflow
else
    ${PYTHON_PIP_CMD} install "tensorflow==${NG_TF_TF_VERSION}"
fi    

xtime="$(date)"
echo  ' '
echo  "===== Run Klocwork actions for nGraph TensorFlow Bridge at ${xtime} ====="
echo  ' '

set -x  # Turn on tracing

export KLOCWORK_LTOKEN="${kw_ltoken}"
echo "DBG:  KLOCWORK_LTOKEN=[${KLOCWORK_LTOKEN}]"
ls -l "${KLOCWORK_LTOKEN}"
echo "DBG:  Contents of ${KLOCWORK_LTOKEN}:"
cat "${KLOCWORK_LTOKEN}"
echo "DBG:  End of contents"

cd "${bridge_dir}"

mkdir "${bbuild_dir}"
cd "${bbuild_dir}"

cmake "${bridge_dir}"

${kw_bin}/kwinject --update make -j16  2>&1  | tee ${bridge_dir}/kwinject-log.txt

${kw_bin}/kwdeploy sync --url "${NG_TF_KW_SERVER}"  2>&1  | tee ${bridge_dir}/kwdeploy-log.txt

# kwbuildproject returns non-zero if build commands returned non-zero.
# This would kill the automated build, so use || true to keep the build going.
( ${kw_bin}/kwbuildproject --url "${NG_TF_KW_SERVER}/${kw_project}" --incremental --tables-directory KWTABLES kwinject.out  2>&1  | tee ${bridge_dir}/kwbuildproject-log.txt ) || true

${kw_bin}/kwadmin --url "${NG_TF_KW_SERVER}" load "${kw_project}" KWTABLES  2>&1  | tee ${bridge_dir}/kwadmin-load-log.txt

set +x  # Turn off tracing


xtime="$(date)"
echo  ' '
echo  "===== Deactivating the Virtual Environment at ${xtime} ====="
echo  ' '

deactivate

xtime="$(date)"
echo ' '
echo "===== Completed Tensorflow Build and Test at ${xtime} ====="
echo ' '
