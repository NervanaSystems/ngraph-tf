#!/bin/bash
set -u
set -e

echo "****************************************************************************************"
echo "Run TensorFlow bridge <----> NGraph-C++ Pre-Merge CI Tests..."
echo "****************************************************************************************"

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z ${TF_ROOT+x} ]; then
    TF_ROOT="$THIS_SCRIPT_DIR"/../../../tensorflow
fi

if [ ! -e $TF_ROOT ]; then
    echo "TensorFlow installation directory not found: " $TF_ROOT
    exit 1
fi

#===================================================================================================
# Run the test...
#===================================================================================================
echo "Running TensorFlow unit tests"
./gtest_ngtf

pushd python
pytest
popd

echo "Running a quick inference test"
pushd ../../examples/resnet
python tf_cnn_benchmarks.py --model=resnet50  --eval --num_inter_threads=1 \
    --batch_size=1  \
    --train_dir /nfs/fm/disks/aipg_trained_dataset/ngraph_tensorflow/partially_trained/resnet50\
    --data_format NCHW --select_device NGRAPH  --num_epochs=1 
    --data_name=imagenet --data_dir /mnt/data/TF_ImageNet_latest/ --datasets_use_prefetch=False \
    --num_batches=50

popd