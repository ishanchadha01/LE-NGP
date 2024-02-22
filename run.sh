cd /home/chen-lab/ishan/LE-NGP

# conda activate neuralangelo

SEQUENCE=ct1a
SCENE_TYPE=object
DATA_PATH=data/cecum_t1_a

EXPERIMENT=ct1a
GROUP=group1
NAME=ct1a_all
CONFIG=projects/splatting/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
TORCH_CPP_LOG_LEVEL=DEBUG
NCCL_DEBUG=DEBUG
LOGLEVEL=DEBUG
python -m torch.distributed.launch --use-env --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar