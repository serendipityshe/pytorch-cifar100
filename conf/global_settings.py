'''configuration for this project

author serendipityshe
'''

from datetime import datetime

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


CHECKPOINT_PATH = 'checkpoint'

EPOCH = 200
MILESTONES = [60, 120, 160]

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'

TIME_NOW = datetime.now().strftime(DATE_FORMAT)

LOG_DIR = 'runs'

SAVE_EPOCH = 20


