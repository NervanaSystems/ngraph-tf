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
    # First download repo
    command_executor("git clone " + repo)

    # # Apply patch to use ngraph_bridge
    # pwd = os.getcwd()
    # os.chdir(pwd + "/models/")
    # command_executor('git apply ' + pwd + '/ngraph_inference.patch')
    # os.chdir(pwd)


def run_inference(model_name):
    parameters = '[{"model_type" : "Image Recognition", "model_name" : "Inception_v4","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/inception_v4.ckpt"}, {"model_type" : "Image Recognition", "model_name" : "MobileNet_v1","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/mobilenet_v1_1.0_224.ckpt"}, {"model_type" : "Image Recognition", "model_name" : "ResNet50_v1","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/resnet_v1_50.ckpt"}, {"model_type" : "Object Detection", "model_name" : "SSD-MobileNet_v1", "CHECKPOINT" : "/nfs/site/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/ssd_mobilenet_v1_coco_2018_01_28/"}]'
    data = json.loads(parameters)

    pwd = os.getcwd()
    for i, d in enumerate(data):
        if (model_name in data[i]["model_name"] and
                data[i]["model_type"] == "Image Recognition"):
            CHECKPOINT = data[i]["CHECKPOINT"]
            DATASET_DIR = data[i]["DATASET"]
            os.chdir(pwd + "/models/research/slim")
            command_executor("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`")
            command_executor('git apply ' + pwd + '/image_recognition.patch')
            if (model_name == "Inception_v4"):
                command_executor(
                    "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path="
                    + CHECKPOINT + " --dataset_dir=" + DATASET_DIR +
                    " --dataset_name=imagenet --dataset_split_name=validation --model_name=inception_v4"
                )
            if (model_name == "MobileNet_v1"):
                command_executor(
                    "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path="
                    + CHECKPOINT + " --dataset_dir=" + DATASET_DIR +
                    " --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v1"
                )
            if (model_name == "ResNet50_v1"):
                command_executor(
                    "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path="
                    + CHECKPOINT + " --dataset_dir=" + DATASET_DIR +
                    " --dataset_name=imagenet --dataset_split_name=validation --model_name=resnet_v1_50 --labels_offset=1"
                )
            os.chdir(pwd)
        if (model_name in data[i]["model_name"] and
                data[i]["model_type"] == "Object Detection"):
            CHECKPOINT = data[i]["CHECKPOINT"]
            os.chdir(pwd + "/models/research/object_detection")
            command_executor("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`")
            command_executor('git apply ' + pwd + '/object_detection.patch')
            command_executor(
                "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python model_main.py --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco.config --checkpoint_path="
                + CHECKPOINT + "--run_once=True")
            os.chdir(pwd)
        else:
            print(
                "Please give a valid Model name. \nAvailble models are Inception_v4, MobileNet_v1, SSD-MobileNet_v1, ResNet50_v1"
            )

    os.chdir(pwd)


def check_accuracy(model):
    #check if the accuracy of the model inference matches with the published numbers
    accuracy = {"Inception_v4": 80.2, "SSD-MobileNet_v1": 26.3}

    line = sys.stdin.readline()
    while line:
        print(line.rstrip())

    # Look for Accuracy
    is_match = re.search('Accuracy', line)

    for k in accuracy:
        for v in accuracy[k]:
            if is_match in v:
                print(model + "Accuracy matches")
            else:
                print(model + "Accuracy does not match")

    #line = sys.stdin.readline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Accuracy verification for TF models using ngraph.')

    parser.add_argument('--model_name', help='Model name to run inference')

    repo = "https://github.com/tensorflow/models.git"

    cwd = os.getcwd()
    # This script must be run from this location
    assert '/'.join(
        cwd.split('/')[-3:]) == 'ngraph-tf/diagnostics/model_accuracy'

    args = parser.parse_args()
    download_repo(repo)
    run_inference(args.model_name)
