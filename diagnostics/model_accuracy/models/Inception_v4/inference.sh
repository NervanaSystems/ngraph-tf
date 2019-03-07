DATASET_DIR=/mnt/data/TF_ImageNet_latest/
PYTHONPATH=`pwd`:`pwd`/research/slim
CHECKPOINT=/nfs/fm/disks/aipg_trained_models_01/tensorflow/inception-v4/imagenet/

cd research/slim

OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,compact,1,0 \
	python eval_image_classifier.py \
	--alsologtostderr \
	--checkpoint_path=$CHECKPOINT \
	--dataset_dir=${DATASET_DIR} \
	--dataset_name=imagenet \
	--dataset_split_name=validation \
	--model_name=inception_v4 






