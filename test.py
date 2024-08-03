import os 
from conf import settings
from utils import most_recent_folder
import datetime

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
    print(folders)

    # 筛选出非空文件夹
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]

    print(folders)
    # 如果文件夹为空，则返回空字符串
    if len(folders) == 0:
        return ''
        
    # 对文件夹进行排序
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    # 返回最后一个文件夹
    return folders[-1]


# os.path.join(settings.CHECKPOINT_PATH, args.net)

recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, 'resnet50'), fmt = settings.DATE_FORMAT)

# print('recent_folder: ', recent_folder) # 2022-01-01

# print(os.path.join(settings.CHECKPOINT_PATH, 'resnet50')) # 2022-01-01
# print(settings.DATE_FORMAT) # %Y-%m-%d
# folders = os.listdir(os.path.join(settings.CHECKPOINT_PATH, 'resnet50'))

# folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, settings.DATE_FORMAT))

# print(folders[-1])