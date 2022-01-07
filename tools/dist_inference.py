import os
import argparse
from posixpath import basename
from sys import path
from numpy.random.mtrand import shuffle

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

import numpy as np
import time

from styletransfer import PerceptualLoss
from styletransfer import StyleTransfer
from styletransfer import COCODataset
from utils import create_logger, SubsetRandomSampler
from configs import get_config

from timm.utils import AverageMeter

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def _denormailize_tensor(x, mean, std):
    for i in range(3):
        x[:, i, :, :] *=  std[i]
        x[:, i, :, :] += mean[i]


def parse_option():
    parser = argparse.ArgumentParser('Style Transfer', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')

    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--output_dir', default='outputs', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config

@torch.no_grad()
def main(config):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    # build dataloader

    dataset_val = COCODataset(data_dir=config.DATA.DATA_PATH, is_train=False, 
                                img_size=256, style_path=config.STYLE_IMG)

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    model = StyleTransfer().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    perceptual_loss_fn = PerceptualLoss().cuda().eval()

    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False

    # load parameters
    pretrained_ckpts = os.listdir(os.path.join(config.OUTPUT_DIR, 'ckpt'))
    latest_ckpt = None
    if len(pretrained_ckpts) > 0:
        pretrained_ckpts.sort()
        latest_ckpt = os.path.join(os.path.join(config.OUTPUT_DIR, 'ckpt', pretrained_ckpts[-1]))
    pretrained_path = config.PRETRAINED_PATH or latest_ckpt
    assert pretrained_path is not None
    pretrained_params = torch.load(
        pretrained_path, map_location='cuda:{}'.format(dist.get_rank()))
    model.load_state_dict(pretrained_params['state_dict'])

    logger.info(f"Successfully load pre-trained parameters from {pretrained_path}")

    for data_batch in val_loader:

        y_c, y_s, fp = data_batch['img'], data_batch['style_image'], data_batch['fp']

        y_c, y_s = y_c.cuda(), y_s.cuda()
        y_hat = model(y_c)

        _denormailize_tensor(
            y_hat, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in range(y_c.shape[0]):
            save_image(
                y_hat.clamp(0, 1), 
                fp=os.path.join(config.OUTPUT_DIR, "val", \
                    os.path.basename(fp[i])),
                normalize=True,
                value_range=(0, 1))


if __name__ == '__main__':
    
    _, config = parse_option()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.OPTIMIZER.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_lr = config.TRAIN.OPTIMIZER.BASE_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.OPTIMIZER.BASE_LR = linear_scaled_lr

    config.freeze()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, 'val'), exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT_DIR, dist_rank=dist.get_rank(), name=f"{config.EXP_NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT_DIR, "config.txt")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)