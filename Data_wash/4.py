# train_soc_fnn.py
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import joblib
from model import FNN, Config_data  # 确保存在model.py 文件
# ================== 参数配置区 ==================
class Config(Config_data):
    # 数据参数
    input_features = ['Voltage', 'Current', 'Temperature']  # 输入特征列
    target_feature = 'SOC'  # 目标列
    test_size = 0.2  # 验证集比例

    # # 模型参数
    # input_size = 3  # 输入特征维度
    # hidden_size = 64  # 隐藏层神经元数
    # output_size = 1  # 输出维度
    # num_layers = 3  # 网络深度（包含输出层）
    # dropout_prob = 0.1  # Dropout概率

    # 训练参数
    batch_size = 256  # 批处理大小
    learning_rate = 1e-3
    epochs = 1000  # 最大训练轮数
    patience = 20  # 早停耐心值

    # 存储路径
    model_path = "./checkpoints/best_model.pth"
    scaler_path = "./checkpoints/scaler.save"


# ================== 数据预处理 ==================
class SOCDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_data(csv_path):
    df = pd.read_csv(csv_path).iloc[:, 1:]  # 跳过第一列编号
    df = df.dropna()
    # 基于电池物理特性过滤异常记录
    df = df[(df['Voltage'] > 2) & (df['Voltage'] < 4.5)]
    df = df[(df['Temperature'] > -30) & (df['Temperature'] < 80)]

    X = df[Config.input_features].values
    y = df[Config.target_feature].values.reshape(-1, 1)

    # # 缺失值处理（双重保障）
    # X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    # y = np.nan_to_num(y, nan=0.5)  # SOC缺失设为中间值

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存scaler
    os.makedirs(os.path.dirname(Config.scaler_path), exist_ok=True)
    joblib.dump(scaler, Config.scaler_path)

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=Config.test_size, random_state=42
    )

    # 转换为Tensor
    train_dataset = SOCDataset(torch.FloatTensor(X_train),
                               torch.FloatTensor(y_train))
    val_dataset = SOCDataset(torch.FloatTensor(X_val),
                             torch.FloatTensor(y_val))

    return train_dataset, val_dataset


# ================== 模型定义 ==================
class FNN(nn.Module):
    def __init__(self, config):
        super(FNN, self).__init__()
        layers = []
        input_dim = config.input_size

        # 动态构建隐藏层
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(input_dim, config.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_prob))
            input_dim = config.hidden_size

            # 输出层
        layers.append(nn.Linear(input_dim, config.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    # ================== 训练流程 ==================


def train_model(csv_path):
    # 初始化配置
    config = Config()

    # 加载数据
    train_dataset, val_dataset = load_data(csv_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = FNN(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    best_loss = math.inf
    early_stop_counter = 0

    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                # print(loss)

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # 早停检测
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{config.epochs}  | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")


# ================== 执行训练 ==================
if __name__ == "__main__":
    Data_path=r'C:\Users\admin\Desktop\数据\LG 18650HG2 Li-ion Battery Data\LG 18650HG2 Li-ion Battery Data\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020\merged_data.csv'
    train_model(Data_path)  # 替换为你的CSV文件路径