# -*- coding: utf-8 -*-
"""
电池SOC预测神经网络v2.5 (2025-03-28更新)
核心功能：基于电压、电流、温度数据预测电池SOC
优化特性：
    - 异常数据自动过滤
    - 训练过程稳定性增强
    - 支持无GPU环境大数据训练
"""

# region ##### 基础库导入 #####
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 进度条显示


# endregion

# region ##### 数据预处理模块 #####
class BatteryDataset(Dataset):
    """自定义数据集类
    功能：
        1. 将numpy数组转换为PyTorch张量
        2. 实现数据长度获取和索引访问
    参数：
        features : 经处理的特征数据 (n_samples, 3)
        labels   : 对应SOC标签 (n_samples,)
    """

    def __init__(self, features, labels):
        # 转换为float32类型节省内存，添加维度适配网络输入
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # 增加维度 -> (n,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def safe_preprocess(file_path):
    """安全数据预处理流程
    处理步骤：
        1. 加载原始数据
        2. 物理范围异常值过滤（电压2.5-4.3V，温度-20-80℃）
        3. 缺失值填充（特征列用均值，标签用0.5）
        4. 标准化处理（仅用训练集拟合scaler）
        5. 数据集划分（6:2:2）
    """
    df = pd.read_csv(file_path)

    # 基于电池物理特性过滤异常记录
    df = df[(df['Voltage'] > 2.5) & (df['Voltage'] < 4.3)]
    df = df[(df['Temperature'] > -20) & (df['Temperature'] < 80)]

    # 特征与标签分离 (假设列顺序为编号,电压,电流,温度,SOC)
    features = df.iloc[:, 1:4].values  # 取第2-4列
    labels = df.iloc[:, 4].values  # 第5列为SOC

    # 缺失值处理（双重保障）
    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
    labels = np.nan_to_num(labels, nan=0.5)  # SOC缺失设为中间值

    # 标准化流程（防止数据泄漏的关键步骤）
    scaler = StandardScaler()
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=42)
    scaler.fit(X_train)  # 重要！仅用训练集计算均值和方差

    # 应用标准化到各数据集
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_temp[:len(y_temp) // 2])  # 前50%验证集
    X_test = scaler.transform(X_temp[len(y_temp) // 2:])  # 后50%测试集

    return X_train, X_val, X_test, y_train, y_temp[:len(y_temp) // 2], y_temp[len(y_temp) // 2:], scaler


# endregion

# region ##### 神经网络模型 #####
class RobustFNN(nn.Module):
    """鲁棒前馈神经网络
    结构特点：
        - 使用LeakyReLU防止梯度消失
        - 添加BatchNorm提升训练稳定性
        - 包含Dropout层防止过拟合
    输入维度：3 (电压, 电流, 温度)
    输出维度：1 (SOC)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),  # 输入层 -> 隐藏层1
            nn.LeakyReLU(0.01),  # 负区间保留0.01斜率
            nn.BatchNorm1d(128),  # 批量归一化加速收敛
            nn.Linear(128, 64),  # 隐藏层1 -> 隐藏层2
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),  # 随机丢弃10%神经元
            nn.Linear(64, 32),  # 隐藏层2 -> 隐藏层3
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)  # 输出层
        )

    def forward(self, x):
        return self.net(x)
    # endregion


# region ##### 训练流程 #####
def train_model(file_path):
    """模型训练主函数
    训练策略：
        - 使用HuberLoss增强异常值鲁棒性
        - AdamW优化器带权重衰减
        - 动态学习率调整(ReduceLROnPlateau)
        - 梯度裁剪防止爆炸
        - 早停机制(10轮无改进停止)
    """
    # 数据准备
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = safe_preprocess(file_path)

    # 创建数据加载器
    train_set = BatteryDataset(X_train, y_train)
    val_set = BatteryDataset(X_val, y_val)
    test_set = BatteryDataset(X_test, y_test)

    # 设置加载器参数（根据内存调整batch_size）
    train_loader = DataLoader(train_set,
                              batch_size=1024,  # 较大批量提升训练速度
                              shuffle=True,  # 打乱顺序增强泛化
                              num_workers=2)  # 并行加载进程数
    val_loader = DataLoader(val_set, batch_size=2048)  # 验证/测试用更大批量
    test_loader = DataLoader(test_set, batch_size=2048)

    # 模型初始化
    model = RobustFNN()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4,  # 初始学习率
                                  weight_decay=1e-5)  # L2正则化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)  # 验证损失5轮不降则降低LR
    criterion = nn.HuberLoss()  # 结合MSE和MAE优点

    # 训练监控
    best_loss = np.inf
    early_stop_counter = 0
    history = {'train': [], 'val': []}

    # 训练循环
    for epoch in range(1000):
        # 训练阶段
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()

            # 梯度裁剪防止NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                current_loss = criterion(pred, y).item()
                if np.isnan(current_loss):  # 安全检测
                    print("检测到NaN值，终止训练!")
                    return
                val_loss += current_loss

                # 计算平均损失
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)  # 更新学习率

        print(
            f"Epoch {epoch + 1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停机制
        if avg_val < best_loss:
            best_loss = avg_val
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("↗ 保存最佳模型")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 30:
                print("早停触发")
                break

                # 模型安全加载
    try:
        model.load_state_dict(torch.load('best_model.pth',
                                         weights_only=True))  # 安全模式
        print("成功加载最佳模型")
    except FileNotFoundError:
        print("警告：未找到保存的模型，使用最后权重")

    # 最终测试
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
    print(f"测试集最终损失: {test_loss / len(test_loader):.4f}")


# endregion

if __name__ == "__main__":
    train_model(r'C:\Users\admin\Desktop\数据\LG 18650HG2 Li-ion Battery Data\LG 18650HG2 Li-ion Battery Data\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\merged_data.csv')