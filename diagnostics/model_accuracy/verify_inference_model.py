# Copyright 2019 Intel Corporation
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
# =============================================================================

import pdb
from subprocess import check_output, call, Popen, PIPE
import json, shlex, os, argparse, sys


def parse_json(json_file_name):
    with open(json_file_name) as f:
        return json.load(f)


def command_executor(cmd, verbose=False, msg=None, stdout=None):
    if verbose or msg is not None:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)

    ps = Popen(cmd, stdin=PIPE, shell=True)
    so, se = ps.communicate()
    #TODO: make this function more robust


def download_repo(repo, target_name=None, version='master'):
    # First download to a temp folder
    command_executor("git clone " + repo + " " +
                     ("" if target_name is None else target_name))
    command_executor("git fetch")
    # Next goto this folder nd determine the name of the root folder
    pwd = os.getcwd()
    # Go to the tree
    os.chdir(target_name)
    # checkout the specified branch
    command_executor("git checkout " + version)
    os.chdir(pwd)


def run_inference(model_dir):
    #TODO: assert TF version. Some models may not run on TF1.12 etc
    model_dir = os.path.abspath(model_dir)

    use_ngraph_models_repo = not os.path.isfile(model_dir + '/repo.txt')
    if use_ngraph_models_repo:
        repo_dl_loc = os.path.abspath(
            model_dir + '/../../../..')  #this is path to ngraph-models root
        assert repo_dl_loc.split('/')[-1] == 'ngraph-models'
    else:
        repo_info = [
            line.strip()
            for line in open(model_dir + '/repo.txt').readlines()
            if len(line.strip()) > 0
        ]
        repo_name = repo_info[0]
        repo_version = repo_info[1] if len(repo_info) == 2 else 'master'
        repo_dl_loc = model_dir + '/downloaded_model'
        #TODO: download only when needed?
        download_repo(repo_name, repo_dl_loc, repo_version)

    cwd = os.getcwd()
    os.chdir(repo_dl_loc)
    if os.path.isfile(model_dir + '/getting_repo_ready.sh'):
        command_executor(model_dir + '/getting_repo_ready.sh')

    # To generate the patch use: git diff > ngraph_inference.patch
    if os.path.isfile(model_dir + '/ngraph_inference.patch'):  #/foo/bar/.git
        command_executor('git apply ' + model_dir + '/ngraph_inference.patch')

    #It is assumed that we need to be in the "model repo" for run_inference to run
    #run_inference is written assuming we are currently in the downloaded repo
    for flname in os.listdir(
            model_dir):  # The model folder can have multiple tests
        if flname.startswith('inference') and 'disabled' not in flname:
            command_executor(
                model_dir + '/' + flname, msg="Running test config: " + flname)
    command_executor('git reset --hard')  # remove applied patch (if any)
    # TODO: each inference.sh could have its own ngraph_inference.patch
    os.chdir(cwd)


def check_functional(model_dir):
    #check if there exists a check_functional.sh in the model folder
    #if not, then use run_functional
    pass


# TODO: what of same model but different configs?
# TODO: what if the same repo supports multiple models?

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Accuracy verification for TF models using ngraph. Performs 2 types of tests. A) inference and B) functional'
    )

    parser.add_argument(
        '--inference',
        action='store_true',
        help='perform inference using ngraph (inference)')
    parser.add_argument(
        '--functional',
        action='store_true',
        help='perform type b tests (functional)')
    parser.add_argument(
        '--models',
        action='store',
        type=str,
        help='comma separated list of model names',
        default='')

    cwd = os.getcwd()
    # This script must be run from this location
    assert '/'.join(
        cwd.split('/')[-3:]) == 'ngraph-tf/diagnostics/model_accuracy'

    args = parser.parse_args()

    if not (args.inference or args.functional):
        print(
            "No type of test enabled. Please choose --inference, --functional or both"
        )
        sys.exit(0)

    model_list = os.listdir(
        'models') if args.models == '' else args.models.split(',')

    for model_name in model_list:
        print('Testing model: ' + model_name)
        if args.inference:
            run_inference('./models/' + model_name)
        if args.functional:
            print('Functional tests not implemented yet!!')

# Sample run script:
# python test_main.py --inference --models Inception_v4
