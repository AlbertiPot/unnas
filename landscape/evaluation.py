"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import sys
sys.path.insert(0,"/data/gbc/workspace/unnas")
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable

def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()

    #########################################################
    # for evaled nets
    with torch.no_grad():
        assert isinstance(criterion, nn.CrossEntropyLoss)

        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            net.drop_path_prob = 0.2

            outputs, logits_aux = net(inputs)
            loss = criterion(outputs, targets)
            if logits_aux is not None:
                loss_aux = criterion(logits_aux, targets)
                loss += loss_aux * 0.4
            total_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
    
    #########################################################
    # original
    # with torch.no_grad():
    #     if isinstance(criterion, nn.CrossEntropyLoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)
    #             targets = Variable(targets)
    #             if use_cuda:
    #                 inputs, targets = inputs.cuda(), targets.cuda()
    #             outputs = net(inputs)
    #             loss = criterion(outputs, targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.eq(targets).sum().item()

    #     elif isinstance(criterion, nn.MSELoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)

    #             one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
    #             one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
    #             one_hot_targets = one_hot_targets.float()
    #             one_hot_targets = Variable(one_hot_targets)
    #             if use_cuda:
    #                 inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
    #             outputs = F.softmax(net(inputs))
    #             loss = criterion(outputs, one_hot_targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.cpu().eq(targets).sum().item()

    # return total_loss/total, 100.*correct/total
