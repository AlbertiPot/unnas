import os
import cifar10.model_loader

def load(dataset, model_file, data_parallel=False):
    if dataset == 'cifar10':
        # net = cifar10.model_loader.load(model_name, model_file, data_parallel)

        # for unnas_model
        net = cifar10.model_loader.load_unnas_evaled_nets(model_file)


    return net
