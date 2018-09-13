#!/bin/bash

# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# ******************************************************************************

#===================================================================================================
# Provides Bash functions for dealing with clang-format.
#===================================================================================================

declare _intelnervana_clang_format_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_clang_format_lib_SCRIPT_NAME}") && pwd )"

source "${_maint_SCRIPT_DIR}/bash_lib.sh"

format_lib_verify_version() {
    if (( $# != 3 )); then
        bash_lib_print_error "Usage: ${FUNCNAME[0]} <clang-format-prog-pathname> <required-version-number> <CLANG or YAPF>"
        return 1
    fi

    local PROGNAME="${1}"
    local REQUIRED_VERSION="${2}"
    local CLANG_OR_YAPF="${3}"


    if ! [[ "${REQUIRED_VERSION}" =~ ^[0-9]+.[0-9]+$ ]]; then
        bash_lib_print_error "${FUNCNAME[0]}: required-version-number must have the form (number).(number)."
        return 1
    fi
    

    if ! [[ -f "${PROGNAME}" ]]; then
        bash_lib_print_error "Unable to find clang-format program named '${PROGNAME}'"
        return 1
    fi

    local VERSION_LINE
    if ! VERSION_LINE=$("${PROGNAME}" --version); then
        bash_lib_print_error "Failed invocation of command '${PROGNAME} --version'"
        return 1
    fi

    local SED_FLAGS
    if [[ "$(uname)" == 'Darwin' ]]; then
        SED_FLAGS='-En'
    else
        SED_FLAGS='-rn'
    fi

    local VERSION_X_Y
    if [[ "${CLANG_OR_YAPF}" =~ "CLANG" ]]; then
        if ! VERSION_X_Y=$(echo "${VERSION_LINE}" | sed ${SED_FLAGS} 's/^clang-format version ([0-9]+.[0-9]+).*$/\1/p')
        then
            bash_lib_print_error "Failed invocation of sed to find clang verion."
            return 1
        fi
    else
        local PYTHON_VERSION_LINE
        if ! PYTHON_VERSION_LINE=$(python --version); then
            bash_lib_print_error "Failed invocation of command 'python --version'"
            return 1
        fi
        echo $PYTHON_VERSION_LINE
        if PYTHON_VERSION=$(echo "${PYTHON_VERSION_LINE}" | sed ${SED_FLAGS} 's/^Python ([0-9]+).*$/\1/p')
        then
            if [[ "3" != "${PYTHON_VERSION}" ]]; then
                bash_lib_print_error "Python reports version number '${PYTHON_VERSION}' but we require 3"
                return 1
            fi
        else
            bash_lib_print_error "Failed invocation of sed to find Python version."
            return 1
        fi
        if ! VERSION_X_Y=$(echo "${VERSION_LINE}" | sed ${SED_FLAGS} 's/^yapf ([0-9]+.[0-9]+).*$/\1/p')
        then
            bash_lib_print_error "Failed invocation of sed to find yapf version."
            return 1
        fi
    fi

    if [[ "${REQUIRED_VERSION}" != "${VERSION_X_Y}" ]]; then
        bash_lib_print_error \
            "Program '${PROGNAME}' reports version number '${VERSION_X_Y}'" \
            "but we require '${REQUIRED_VERSION}'"
        return 1
    fi

}

