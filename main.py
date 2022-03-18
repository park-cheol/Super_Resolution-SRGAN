import argparse
import warnings
import os
import random
import time
import numpy as np
import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from multiprocessing import Pool
from torch.autograd import Variable

from models.Discriminator import Discriminator
from models.Generator import Generator
from models.FeatureExtractor import FeatureExtractor
from data.dataset import ImageDataset
from utils.utils import sample


parser = argparse.ArgumentParser()

parser.add_argument('--data', metavar='DIR', default='./dataset/img_align_celeba', help='path to dataset (default: imagenet)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='Number of max epochs in training (default: 100)')
parser.add_argument('--decay-epoch', type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--sample-freq', type=int, default=10000)

# SRGAN
parser.add_argument('--lr', default=2e-4)
parser.add_argument('--hr-height', type=int, default=256, help="high res. image height")
parser.add_argument('--hr-width', type=int, default=256, help="high res. image width")
parser.add_argument('--channels', type=int, default=3)

# mode
parser.add_argument('--evaluate', '-e', default=False, action='store_true')
parser.add_argument('--generate', '-g', default=False, action='store_true')
parser.add_argument('--generate-file', type=str, help="generate 하고 싶은 파일경로")

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()
    os.makedirs('saved_models', exist_ok=True)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    generator = Generator(in_channels=args.channels, out_channels=args.channels)
    discriminator = Discriminator()
    feature_extractor = FeatureExtractor()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            generator.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            feature_extractor.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
            feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[args.gpu])
        else:
            generator.cuda()
            discriminator.cuda()
            feature_extractor.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            generator = torch.nn.parallel.DistributedDataParallel(generator)
            discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)
            feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = generator.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        feature_extractor = feature_extractor.cuda(args.gpu)
    else:
        generator = torch.nn.DataParallel(generator).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()
        feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()

    # Criterion / Optimizer
    criterion_GAN = nn.MSELoss().cuda(args.gpu)
    criterion_content = nn.L1Loss().cuda(args.gpu)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr,
                                           betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr,
                                               betas=(0.5, 0.999))

    # load model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            generator.load_state_dict(checkpoint['G'])
            discriminator.load_state_dict(checkpoint['D'])
            generator_optimizer.load_state_dict(checkpoint['G_optimizer'])
            discriminator_optimizer.load_state_dict(checkpoint['D_optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Dataset / DataLoader
    train_dataset = ImageDataset(root=args.data, hr_shape=(args.hr_height, args.hr_width))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    feature_extractor.eval()  # freeze vgg
    G_param = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    D_param = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    # f_param = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    print("g_param", G_param)
    print("d_param", D_param)
    # print("f_param", f_param)
    print("total: ", G_param + D_param)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, generator, discriminator, feature_extractor,
              criterion_GAN, criterion_content,
              generator_optimizer, discriminator_optimizer,
              epoch, summary, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save({
                'epoch': epoch + 1,
                'G': generator.state_dict(),
                'D': discriminator.state_dict(),
                'G_optimizer': generator_optimizer.state_dict(),
                'D_optimizer': discriminator_optimizer.state_dict()
            }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(train_loader, generator, discriminator, feature_extractor,
          criterion_GAN, criterion_content,
          generator_optimizer, discriminator_optimizer,
          epoch, summary, args):
    end = time.time()

    for i, (img_lr, img_hr) in enumerate(train_loader):
        generator.train()
        discriminator.train()

        img_lr = img_lr.cuda(args.gpu, non_blocking=True)
        img_hr = img_hr.cuda(args.gpu, non_blocking=True)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img_lr.size(0), *(1, 16, 16)))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img_lr.size(0), *(1, 16, 16)))), requires_grad=False)

        # Generator Update
        fake_hr = generator(img_lr)
        fake_features = feature_extractor(fake_hr)
        real_features = feature_extractor(img_hr)

        loss_GAN = criterion_GAN(discriminator(fake_hr), valid)
        loss_content = criterion_content(fake_features, real_features.detach())

        generator_loss = loss_content + 1e-3 * loss_GAN

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # Discriminator Update
        loss_real = criterion_GAN(discriminator(img_hr), valid)
        loss_fake = criterion_GAN(discriminator(fake_hr.detach()), fake)
        discriminator_loss = (loss_real + loss_fake) / 2

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        niter = epoch * len(train_loader) + i
        summary.add_scalar('Train/G_loss', generator_loss.item(), niter)
        summary.add_scalar('Train/D_loss', discriminator_loss.item(), niter)
        summary.add_scalar('Train/content_loss', loss_content.item(), niter)
        summary.add_scalar('Train/gan_loss', (loss_GAN * 1e-3).item(), niter)
        summary.add_scalar('Train/D_real', loss_real.item(), niter)
        summary.add_scalar('Train/D_fake', loss_fake.item(), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | D_loss: %f | D_fake: %f | D_real: %f | G_loss: %f | G_Adv: %f | G_cont: %f"
                  % (epoch + 1, i, len(train_loader), discriminator_loss, loss_fake, loss_real, generator_loss,
                     loss_GAN*1e-3, loss_content))

        if i % args.sample_freq == 0:
            sample(niter, img_lr, img_hr, generator)

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


if __name__ == "__main__":
    main()
