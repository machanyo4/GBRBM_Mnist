from model import GBRBM
from dataset import ExtendedMNISTDatasetOnehot, ExtendedMNISTDatasetZeros, k_sampling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.metrics import accuracy_score
import numpy as np

# Seed の固定
generator = torch.Generator()
generator.manual_seed(0)

# GPUが利用可能か確認
print("GPU available : ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの変換
transform = transforms.Compose([transforms.ToTensor()])

# MNISTデータセットのダウンロード
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Validation の作成
train_size = int(0.8 * len(train_dataset))
train_val_size = len(train_dataset) - train_size
test_size = int(0.8 * len(test_dataset))
test_val_size = len(test_dataset) - test_size
train_dataset, train_val_dataset = torch.utils.data.random_split(train_dataset, [train_size, train_val_size], generator=generator)
test_dataset, test_val_dataset = torch.utils.data.random_split(test_dataset, [test_size, test_val_size], generator=generator)

# 拡張データセットの作成
train_dataset = ExtendedMNISTDatasetOnehot(train_dataset)
train_val_dataset = ExtendedMNISTDatasetOnehot(train_val_dataset)
test_dataset = ExtendedMNISTDatasetOnehot(test_dataset)
test_val_dataset = ExtendedMNISTDatasetZeros(test_val_dataset)

# データローダーの作成
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
train_val_loader = torch.utils.data.DataLoader(dataset=train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
test_val_loader = torch.utils.data.DataLoader(dataset=test_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

# モデルの定義
visible_units = 28 * 28 +10  # Mnist画像サイズ
hidden_units = 4096  # 隠れ層のユニット数
# モデルの初期化
rbm = GBRBM(visible_units, hidden_units).to(device)

# パラメータの挿入
model_path = './evaluation/model.pth'
rbm.load_state_dict(torch.load(model_path))
print('Complete parameter adaptation.')

# ---- label included -----
# rbm.eval()
# data = next(iter(train_loader))
# bind_data = data.view(data.size(0), -1).to(device)
# predict_label = rbm.backward(rbm(bind_data))[:, 28*28:]
# correct_label = bind_data[:, 28*28:]
# for i in range(batch_size):
#     print('correct', correct_label.cpu().numpy()[i])
#     print('pred   ', np.round(predict_label.cpu().detach().numpy()[i], decimals=3))
#     print('-----------------------------------------')

# ---- zero label ----
# rbm.eval()
# zeros_label = torch.zeros(batch_size,10).to(device)
# data = next(iter(train_val_loader))
# bind_data = data.view(data.size(0), -1).to(device)
# img_data = bind_data[:, :28*28]
# zeros_data = torch.cat((img_data, zeros_label), dim=1)
# predict_label = k_sampling(10, rbm, zeros_data)[:, 28*28:]
# correct_label = bind_data[:, 28*28:]
# for i in range(batch_size):
#     print('correct', correct_label.cpu().numpy()[i])
#     print('pred   ', np.round(predict_label.cpu().detach().numpy()[i], decimals=3))
#     print('-----------------------------------------')

# ---- rand label ----
rbm.eval()
rand_label = torch.randn(batch_size, 10).to(device)
data = next(iter(train_val_loader))
bind_data = data.view(data.size(0), -1).to(device)
img_data = bind_data[:, :28*28]
zeros_data = torch.cat((img_data, rand_label), dim=1)
predict_label = k_sampling(100, rbm, zeros_data)[:, 28*28:]
correct_label = bind_data[:, 28*28:]
# print(predict_label.cpu().detach().numpy())
for i in range(batch_size):
    print('correct', correct_label.cpu().numpy()[i])
    print('randm ', np.round((rand_label.cpu().numpy()[1]), decimals=2))
    print('pred   ', np.round(predict_label.cpu().detach().numpy()[i]))
    print('-----------------------------------------')