import sys



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

def get_training_dataloader():
    '''获得训练数据集
    Args:
        mean: cifar100的均值
        std: cifar100的标准差
        path: 训练集数据的路径
        batch_size
        num_workers
        shuffle:
    Returns: 
    '''

def get_test_dataloader():
    pass

def compute_mean_std():
    pass

class WarmUpLR():
    pass

def most_recent_folder():
    pass

def most_recent_weights():
    pass

def last_epoch():
    pass

def best_acc_weights():
    pass