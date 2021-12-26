import os
import argparse
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

def _print_log(batch_time, data_time, loss_dict, current_iter, total_iters):
    _ret_string = f"[Iter {current_iter:06d}|{total_iters:06d}] "
    eta = (total_iters - current_iter) * batch_time
    m, s = divmod(eta, 60)
    h, m = divmod(m, 60)
    d, m = divmod(h, 24)
    _ret_string += f"eta: {d} day:{int(h):02d}:{int(m):02d}:{int(s):02d} "
    _ret_string += f"batch time: {batch_time:.03f} data time: {data_time:.03f} "
    for key in loss_dict.keys():
        _ret_string += f"{key}: {loss_dict[key]:.03f} "
    
    return _ret_string

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

def main(config):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    # build dataloader
    dataset_train = COCODataset(data_dir=config.DATA.DATA_PATH, is_train=True, 
                                img_size=256, style_path=config.STYLE_IMG)
    
    dataset_val = COCODataset(data_dir=config.DATA.DATA_PATH, is_train=False, 
                                img_size=256, style_path=config.STYLE_IMG)
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, # sampler=sampler_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    train_loader = sample_data(train_loader)

    START_ITER = 0
    _iter = START_ITER

    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()

    model = StyleTransfer().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    perceptual_loss_fn = PerceptualLoss().cuda().eval()

    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 
         'lr': config.TRAIN.OPTIMIZER.BASE_LR},
    ])

    lambda_style = 1e10
    lambda_feat  = 1e5
    lambda_l2    = 0

    _data_start = time.time()
    for y_c, y_s in train_loader:
        data_time_meter.update(time.time() - _data_start)
        loss_dict = {}
        _iter += 1

        y_c, y_s = y_c.cuda(), y_s.cuda()
        y_hat = model(y_c)
        style_loss, feat_loss = perceptual_loss_fn(y_s, y_hat, y_c)
        l2_loss = torch.nn.functional.mse_loss(y_hat, y_c)

        loss = style_loss * lambda_style \
             + feat_loss * lambda_feat \
             + l2_loss * lambda_l2

        loss_dict = {
            'style_loss': style_loss.cpu().data * lambda_style,
            'feat_loss': feat_loss.cpu().data * lambda_feat,
            'l2_loss': l2_loss.cpu().data * lambda_l2,
            'loss': loss.cpu().data,
        }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_meter.update(time.time() - _data_start)
        if _iter % config.PRINT_FREQ == 0:
            _ret_string = _print_log(batch_time_meter.avg, data_time_meter.avg, \
                loss_dict, _iter, config.TRAIN.TOTAL_ITERS)
            _denormailize_tensor(
                y_hat, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            save_image(
                y_hat.clamp(0, 1), 
                fp=os.path.join(config.OUTPUT_DIR, f"Iter_{_iter:06d}_{global_rank}.png"),
                normalize=True,
                value_range=(0, 1))
            logger.info(_ret_string)

        if _iter > config.TRAIN.TOTAL_ITERS:
            break
        
        _data_start = time.time()

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
    logger = create_logger(output_dir=config.OUTPUT_DIR, dist_rank=dist.get_rank(), name=f"{config.EXP_NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT_DIR, "config.txt")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)