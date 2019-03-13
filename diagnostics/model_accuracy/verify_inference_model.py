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
from subprocess import check_output, call, Popen, PIPE, STDOUT
import re
import json, shlex, os, argparse, sys


def parse_json(json_file_name):
    with open(json_file_name) as f:
        return json.load(f)


def command_executor(cmd, verbose=True, msg=None):
    if verbose or msg is not None:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)

    ps = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    so, se = ps.communicate()
    return so
    #TODO: make this function more robust


def download_repo(repo, target_name=None, version='master'):
    # First download repo
    command_executor("git clone " + repo)


def run_inference(model_name):
    parameters = '[{"model_type" : "Image Recognition", "model_name" : "Inception_v4","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/inception_v4.ckpt"}, {"model_type" : "Image Recognition", "model_name" : "MobileNet_v1","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/mobilenet_v1_1.0_224.ckpt"}, {"model_type" : "Image Recognition", "model_name" : "ResNet50_v1","DATASET" : "/mnt/data/TF_ImageNet_latest/","CHECKPOINT" : "/nfs/site/home/skantama/validation/models/research/checkpoints/resnet_v1_50.ckpt"}, {"model_type" : "Object Detection", "model_name" : "SSD-MobileNet_v1", "CHECKPOINT" : "/nfs/site/disks/aipg_trained_dataset/ngraph_tensorflow/fully_trained/ssd_mobilenet_v1_coco_2018_01_28/"}]'

    try:
        data = json.loads(parameters)
    except:
        print("Pass a valid model prameters dictionary")

    repo = "https://github.com/tensorflow/models.git"
    pwd = os.getcwd()

    for i, d in enumerate(data):
        if (model_name in data[i]["model_name"]):
            download_repo(repo)
            if (data[i]["model_type"] == "Image Recognition"):
                CHECKPOINT = data[i]["CHECKPOINT"]
                DATASET_DIR = data[i]["DATASET"]
                os.chdir(pwd + "/models/research/slim")
                command_executor("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`")
                command_executor('git apply ' + pwd +
                                 '/image_recognition.patch')

            #TODO:SSD-MobileNet-v1 is not work, will re-enable this after debugging
            # if(data[i]["model_type"] == "Object Detection"):
            #     CHECKPOINT = data[i]["CHECKPOINT"]
            #     command_executor('git apply ' + pwd + '/object_detection.patch')
            #     os.chdir(pwd + "/models/research/")
            #     command_executor("python setup.py install")
            #     command_executor("python slim/setup.py install")
            #     command_executor("export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/object_detection:`pwd`/slim")
    try:
        if (model_name in "Inception_v4"):
            cmd = "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path=" + CHECKPOINT + " --dataset_dir=" + DATASET_DIR + " --dataset_name=imagenet --dataset_split_name=validation --model_name=inception_v4"
            p = command_executor(cmd)
            os.chdir(pwd)

        if (model_name in "MobileNet_v1"):
            cmd = "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path=" + CHECKPOINT + " --dataset_dir=" + DATASET_DIR + " --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v1"
            p = command_executor(cmd)
            os.chdir(pwd)

        if (model_name in "ResNet50_v1"):
            cmd = "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python eval_image_classifier.py --alsologtostderr --checkpoint_path=" + CHECKPOINT + " --dataset_dir=" + DATASET_DIR + " --dataset_name=imagenet --dataset_split_name=validation --model_name=resnet_v1_50 --labels_offset=1"
            p = command_executor(cmd)
            os.chdir(pwd)

        #TODO:SSD-MobileNet_v1 does not run using ngraph, need to debug.
        # if (model_name in "SSD-MobileNet_v1"):
        #     command_executor(
        #         "OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 python object_detection/model_main.py --logtostderr --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_coco.config --checkpoint_dir="
        #         + CHECKPOINT + "--run_once=True")
        #     os.chdir(pwd)

        return model_name, p
    except:
        print(
            "Please pass a valid Model name. Availble models are Inception_v4, MobileNet_v1, ResNet50_v1"
        )
        sys.exit(0)


def check_accuracy(model, p):
    #check if the accuracy of the model inference matches with the published numbers
    accuracy = '[{"model_name" : "Inception_v4", "accuracy" : "0.95194"}, {"model_name" : "ResNet50_v1", "accuracy" : "0.752"}, {"model_name" : "MobileNet_v1", "accuracy" : "0.71018"}]'
    data = json.loads(accuracy)

    for line in p.splitlines():
        print(line)
        if ('eval/Accuracy'.encode() in line):
            top1_accuracy = re.search("\[(.*?)\]", line.decode()).group(1)
        #for now we just validate top 1 accuracy, but calculating top5 anyway.
        if ('eval/Recall_5'.encode() in line):
            top5_accuracy = float(
                re.search("\[(.*?)\]", line.decode()).group(1))

    for i, d in enumerate(data):
        if (model in data[i]["model_name"]):
            # Tolerance check
            diff = abs(float(top1_accuracy) - float(data[i]["accuracy"]))
            if (diff <= 0.001):
                print("\nResult:Model Accuracy is expected for " + model + ' ' +
                      top1_accuracy)
            else:
                print("\nResult:Model Accuracy is not as expected for " +
                      model + ' ' + top1_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Accuracy verification for TF models using ngraph.')

    parser.add_argument('--model_name', help='Model name to run inference')

    cwd = os.getcwd()
    # This script must be run from this location
    assert '/'.join(
        cwd.split('/')[-3:]) == 'ngraph-tf/diagnostics/model_accuracy'

    args = parser.parse_args()

    #Just takes in one model at a time for now
    #TODO(Sindhu): Run multiple or ALL models at once and compare accuracy.

    model_name, p = run_inference(args.model_name)
    check_accuracy(model_name, p)
