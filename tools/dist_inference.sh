CONFIG=styletransfer_vgg16_rain_princess
GPUS=2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node $GPUS \
    --master_port 12345 $(dirname "$0")/dist_inference.py \
    --cfg configs/$CONFIG.yaml