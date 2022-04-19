#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os
from thop import profile

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
import torch.nn.functional as F
from pycls.core.config import cfg


logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config 保存config.yml文件
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()  # 根据全局的yacs config返回一个网络的实例，网络的实例根据config中的MODEL.TYPE
    logger.info("Model:\n{}".format(model))
    
    # Log model complexity
    # logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    
    # cityscapes分割的分辨率是1025*2049
    if cfg.TASK == "seg" and cfg.TRAIN.DATASET == "cityscapes":
        h, w = 1025, 2049
    else:
        h, w = cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE
    
    # 制作input，为thop的profile统计FLOPs和参数量做输入
    if cfg.TASK == "jig":
        x = torch.randn(1, cfg.JIGSAW_GRID ** 2, cfg.MODEL.INPUT_CHANNELS, h, w) # 对于jigsaw，是 (1,9,3,h,w)的tensor
    else:
        x = torch.randn(1, cfg.MODEL.INPUT_CHANNELS, h, w)      # 对于其他任务，是(1,3,h,w)
    macs, params = profile(model, inputs=(x, ), verbose=False)
    logger.info("Params: {:,}".format(params))
    logger.info("Flops: {:,}".format(macs))
    
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device  # 一个进程管理一个模型的创建和模型在GPU上的运算，将本进程内初始化后的模型用DDP包裹后，成为一个DDP的副本
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # Set complexity function to be module's complexity function
        # model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Update drop path prob for NAS
    if cfg.MODEL.TYPE == "nas":
        m = model.module if cfg.NUM_GPUS > 1 else model
        m.set_drop_path_prob(cfg.NAS.DROP_PROB * cur_epoch / cfg.OPTIM.MAX_EPOCH)
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate per epoch
    if not cfg.OPTIM.ITER_LR:
        lr = optim.get_epoch_lr(cur_epoch)
        optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Update the learning rate per iter
        if cfg.OPTIM.ITER_LR:
            lr = optim.get_epoch_lr(cur_epoch + cur_iter / len(train_loader))
            optim.set_lr(optimizer, lr)
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        if isinstance(preds, tuple):
            loss = loss_fun(preds[0], labels) + cfg.NAS.AUX_WEIGHT * loss_fun(preds[1], labels)
            preds = preds[0]
        else:
            loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size) # 将所有进程的结果更新[这里相当于所有进程保存了一个结果副本，就不能直接放到主进程吗]
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def search_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of differentiable architecture search. 搜索darts结构"""
    
    m = model.module if cfg.NUM_GPUS > 1 else model     # 分布式下，DDP副本module为m
    # Shuffle the data
    loader.shuffle(train_loader[0], cur_epoch)
    loader.shuffle(train_loader[1], cur_epoch)
    
    # Update the learning rate per epoch
    if not cfg.OPTIM.ITER_LR:
        lr = optim.get_epoch_lr(cur_epoch)
        optim.set_lr(optimizer[0], lr)
    
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    trainB_iter = iter(train_loader[1])     # 优化结构参数的dataloader
    for cur_iter, (inputs, labels) in enumerate(train_loader[0]):
        # Update the learning rate per iter
        if cfg.OPTIM.ITER_LR:
            lr = optim.get_epoch_lr(cur_epoch + cur_iter / len(train_loader[0]))    # 根据指定的lrschedule(cos,exp或steps)获得lr
            optim.set_lr(optimizer[0], lr)
        
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)  
       
        # Update architecture 开始用另一半数据训练结构参数
        if cur_epoch + cur_iter / len(train_loader[0]) >= cfg.OPTIM.ARCH_EPOCH: # cfg.OPTIM.ARCH_EPOCH = 1，第一个epoch更新权重w，只有第二个epoch才开始更新α
            try:
                inputsB, labelsB = next(trainB_iter)
            except StopIteration:
                trainB_iter = iter(train_loader[1])
                inputsB, labelsB = next(trainB_iter)
            inputsB, labelsB = inputsB.cuda(), labelsB.cuda(non_blocking=True)
            
            optimizer[1].zero_grad()        # optimizer1 负责优化结构参数α
            loss = m._loss(inputsB, labelsB)# 在训练集的rside前向一次，计算loss，!!!!这里仅仅是一阶更新
            loss.backward()
            optimizer[1].step()             # 更新结构参数
        
        # Perform the forward pass 开始更新权重
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer[0].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # Update the parameters
        optimizer[0].step()
        
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)           # [2, 313, 224, 224] 第二维的量化颜色分类预测换到最后一个维度上 
            preds = preds.reshape(-1, preds.size(3))    # [100352, 313]
            labels = labels.reshape(-1)                 # [2, 224, 224] → [100352]
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])   # 每一个进程都会得到这三个的平均结果
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()         # 在meters中计算iou用了torch，仍在gpu上
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)  # 本进程若是主进程，log信息，非主进程logger设置为不打印
    train_meter.reset()
    # Log genotype
    genotype = m.genotype()
    logger.info("genotype = %s", genotype)
    logger.info(F.softmax(m.net_.alphas_normal, dim=-1))
    logger.info(F.softmax(m.net_.alphas_reduce, dim=-1))


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        if cfg.TASK == "col":
            preds = preds.permute(0, 2, 3, 1)
            preds = preds.reshape(-1, preds.size(3))
            labels = labels.reshape(-1)
            mb_size = inputs.size(0) * inputs.size(2) * inputs.size(3) * cfg.NUM_GPUS
        else:
            mb_size = inputs.size(0) * cfg.NUM_GPUS
        if cfg.TASK == "seg":
            # top1_err is in fact inter; top5_err is in fact union
            top1_err, top5_err = meters.inter_union(preds, labels, cfg.MODEL.NUM_CLASSES)
        else:
            ks = [1, min(5, cfg.MODEL.NUM_CLASSES)]  # rot only has 4 classes
            top1_err, top5_err = meters.topk_errors(preds, labels, ks)
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        if cfg.TASK == "seg":
            top1_err, top5_err = top1_err.cpu().numpy(), top5_err.cpu().numpy()
        else:
            top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, mb_size)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def train_model():
    """Trains the model."""
    # Setup training/testing environment 设置log，随机数，cudnn backend
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()   # 获得的是分布式模型的在本进程内的一个DDP 副本
    loss_fun = builders.build_loss_fun().cuda()
    
    # 设置被搜索的参数
    # if上分支是darts搜索时的优化器构建部分，else分支是darts做eval的
    if "search" in cfg.MODEL.TYPE:
        params_w = [v for k, v in model.named_parameters() if "alphas" not in k]    # 算子权重参数
        params_a = [v for k, v in model.named_parameters() if "alphas" in k]        # 算子的结构参数
        
        # 权重优化器
        optimizer_w = torch.optim.SGD(
            params=params_w,
            lr=cfg.OPTIM.BASE_LR,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV
        )
        # 结构参数优化器
        if cfg.OPTIM.ARCH_OPTIM == "adam":
            optimizer_a = torch.optim.Adam(
                params=params_a,
                lr=cfg.OPTIM.ARCH_BASE_LR,
                betas=(0.5, 0.999),
                weight_decay=cfg.OPTIM.ARCH_WEIGHT_DECAY
            )
        elif cfg.OPTIM.ARCH_OPTIM == "sgd":
            optimizer_a = torch.optim.SGD(
                params=params_a,
                lr=cfg.OPTIM.ARCH_BASE_LR,
                momentum=cfg.OPTIM.MOMENTUM,
                weight_decay=cfg.OPTIM.ARCH_WEIGHT_DECAY,
                dampening=cfg.OPTIM.DAMPENING,
                nesterov=cfg.OPTIM.NESTEROV
            )
        optimizer = [optimizer_w, optimizer_a]  # [权重优化器，结构参数优化器]
    else:
        optimizer = optim.construct_optimizer(model)
    
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)    # 将ckp中的模型参数，优化器权重读取，返回当前的epoch值
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    
    # Create data loaders and meters
    if cfg.TRAIN.PORTION < 1:   # 自定义数据集切割比例, 若不对训练接进一步切割，config中默认是1，直接跳过
        if "search" in cfg.MODEL.TYPE:
            train_loader = [loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="l"    # 训练集按照portion分割的左部
            ),
            loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="r"    # 训练集按照portion分割的右部
            )]
        else:
            train_loader = loader._construct_loader(
                dataset_name=cfg.TRAIN.DATASET,
                split=cfg.TRAIN.SPLIT,
                batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                shuffle=True,
                drop_last=True,
                portion=cfg.TRAIN.PORTION,
                side="l"
            )
        test_loader = loader._construct_loader(
            dataset_name=cfg.TRAIN.DATASET,
            split=cfg.TRAIN.SPLIT,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
            portion=cfg.TRAIN.PORTION,
            side="r"
        )
    else:
        train_loader = loader.construct_train_loader()
        test_loader = loader.construct_test_loader()
    
    train_meter_type = meters.TrainMeterIoU if cfg.TASK == "seg" else meters.TrainMeter # 分割任务是带IoU，分类任务是top1和top5 acc
    test_meter_type = meters.TestMeterIoU if cfg.TASK == "seg" else meters.TestMeter
    
    l = train_loader[0] if isinstance(train_loader, list) else train_loader
    train_meter = train_meter_type(len(l))  # len(l) = len(data_loader) = No. batchsize
    test_meter = test_meter_type(len(test_loader))
    
    # Compute model and loader timings
    # if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
    #     l = train_loader[0] if isinstance(train_loader, list) else train_loader
    #     benchmark.compute_time_full(model, loss_fun, l, test_loader)    # 跑一下计算和dataloader的时间
    
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        
        # Train for one epoch
        f = search_epoch if "search" in cfg.MODEL.TYPE else train_epoch     # search_epoch是搜索darts结构
        f(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
