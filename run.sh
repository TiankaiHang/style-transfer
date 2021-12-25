CONFIG=styletransfer_vgg16_rain_princess
GPUS=2

python -m torch.distributed.launch --nproc_per_node $GPUS \
    --master_port 12345 train.py \
    --cfg configs/$CONFIG.yaml