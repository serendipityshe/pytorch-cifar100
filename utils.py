import os
import sys
import numpy as np
import datetime
import re

from torch.optim.optimizer import Optimizer
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torchvision
from torch.optim.lr_scheduler import _LRScheduler

def get_network(args):

    '''
    获取网络
    '''

    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    else: 
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net

def get_training_dataloader(mean, std, batch_size = 16, num_workers = 2, shuffle = True):
    '''获得训练数据集
    Args:
        mean: cifar100的均值
        std: cifar100的标准差
        path: 训练集数据的路径
        batch_size: dataloader batchsize
        num_workers: dataloader num_workers
        shuffle: 是否打乱
    Returns: train_data_loader:torch dataloader object
    '''

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    cifar100_training = torchvision.datasets.CIFAR100(root = './data', train= True, download= True, transform= transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle= shuffle, num_workers= num_workers, batch_size= batch_size
    )

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size = 16, num_workers = 2, shuffle = False):
    '''获得测试集
    Args:
        mean: cifar100的均值
        std: cifar100的标准差
        batch_size: dataloader batchsize
        num_workers: dataloader num_workers
        shuffle: 是否打乱
    Returns: test_data_loader: torch dataloader object
    '''


    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    cifar100_test = torchvision.datasets.CIFAR100(root= './data', train= False, download= True, transform= transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, batch_size= batch_size, shuffle= shuffle, num_workers= num_workers
    )

    return cifar100_test_loader

def compute_mean_std(cifar100_datasset):
    '''计算训练集的均值和标准差
    Args:
        cifar100_datasset: cifar100 dataset
    Returns: mean: cifar100的均值, std: cifar100的标准差
    '''
    data_r = np.dstack([cifar100_datasset[i][1][:, :, 0] for i in range(len(cifar100_datasset))])
    data_g = np.dstack([cifar100_datasset[i][1][:, :, 1] for i in range(len(cifar100_datasset))])
    data_b = np.dstack([cifar100_datasset[i][1][:, :, 2] for i in range(len(cifar100_datasset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std 

class WarmUpLR(_LRScheduler):
    '''warmup学习率策略
    Args:
        optimizer: 优化器
        total_iters: warmup的总迭代数
    '''
    def __init__(self, optimizer, total_iters, last_epoch = -1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        '''
        将学习率设置为初始学习率乘以最后的epoch除以warmup的总迭代数
            base_lr * m / total_iters
        '''
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def most_recent_folder(net_weights, fmt):
    '''获取net_weights文件夹下最新的文件名
    如果文件夹下为空则返回空文件夹
    Args:
        net_weights: net_weights文件夹
        fmt: 文件名格式
    Returns: latest folder: 最新文件夹
    '''
   # 获取net_weights文件夹下的所有文件夹
    folders = os.listdir(net_weights)

    # 筛选出非空文件夹
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    # 如果文件夹为空，则返回空字符串
    if len(folders) == 0:
        return ''
        
    # 对文件夹进行排序
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    # 返回最后一个文件夹
    return folders[-1]

def most_recent_weights(weights_folder):
    '''
    获取最新的权重文件
    如果文件为空，则返回空字符串
    '''

    # 获取权重文件夹中的文件列表
    weight_files = os.listdir(weights_folder)
# 如果文件夹为空，则返回空字符串
    if len(weights_folder) == 0:
        return ''
    
# 正则表达式，用于匹配文件名中的参数
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

# 使用正则表达式对文件进行排序
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

# 返回排序后的最后一个文件名
    return weight_files[-1]

def last_epoch(weights_folder):
    '''
    获取最新的权重文件中的epoch数
    如果文件为空，则返回0
    '''
    # 获取最近的权重文件
    weight_file = most_recent_weights(weights_folder)
    # 如果没有找到最近的权重文件，则抛出异常
    if not weight_file:
        raise Exception('no recent weights were found')
    # 从权重文件名中获取epoch数
    resume_epoch = int(weight_file.split('-')[1])

    # 返回epoch数
    return resume_epoch
    

def best_acc_weights(weights_folder):
    '''获取最优的权重文件
    返回给定文件夹中最佳acc .pth文件,如果没有找到最佳acc权重文件,则返回空字符串
    '''

        # 获取权重文件夹中的文件列表
    files = os.listdir(weights_folder)
    # 如果没有文件，则返回空字符串
    if len(files) == 0:
        return ''
        
    # 定义正则表达式，用于匹配文件名中的参数
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    # 找出所有以best结尾的文件
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    # 如果没有best文件，则返回空字符串
    if len(best_files) == 0:
        return ''
        
    # 对best文件进行排序，按照文件名中的数字进行排序
    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).group()[1]))
    # 返回最后一个best文件
    return best_files[-1]
        