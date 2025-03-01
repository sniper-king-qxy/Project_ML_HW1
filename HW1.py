#数据处理
import math
import random

import numpy as np
#读取写入数据
import pandas as pd
import os
import csv
#进度条
from tqdm import tqdm
#pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
#绘制图像
from torch.utils.tensorboard import SummaryWriter

#设置随机种子
def same_seed(seed):
    torch.backends.mkl.deterministic=True
    torch.backends.mkl.benchmark=False
    np.random.seed(seed)
    torch.manual_seed(seed)

#划分数据集
def train_valid_split(data_set,valid_ratio,seed):
    valid_data_size=int(len(data_set)*valid_ratio)
    train_data_size=len(data_set)-valid_data_size
