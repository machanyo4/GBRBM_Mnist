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
batch_size = 512
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
train_val_loader = torch.utils.data.DataLoader(dataset=train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
test_val_loader = torch.utils.data.DataLoader(dataset=test_val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

# 保存されたパラメータの読み込み
# モデルの定義
visible_units = 28 * 28  # Mnist画像サイズ
hidden_units = 4096  # 隠れ層のユニット数
# モデルの初期化
ex_rbm = GBRBM(visible_units, hidden_units).to(device)
ex_model_path = './extraction/model.pth'
ex_rbm.load_state_dict(torch.load(ex_model_path))
print(f'Model parameters loaded from {ex_model_path}')
print(f'raw W size : {ex_rbm.state_dict()["W"].size()}')
print(f'raw visible_bias size : {ex_rbm.state_dict()["visible_bias"].size()}')
# 各パラメータの変更
rand_W_io = torch.randn(10, hidden_units).to(device)
ex_rbm.W.data = torch.cat([ex_rbm.W.data, rand_W_io], dim=0)
rand_vb_io = torch.randn(10).to(device)
ex_rbm.visible_bias.data = torch.cat([ex_rbm.visible_bias.data, rand_vb_io], dim=0)
print('Parameter reshape successful.')
print(f'new W size : {ex_rbm.state_dict()["W"].size()}')
print(f'new visible_bias size : {ex_rbm.state_dict()["visible_bias"].size()}')

#--------------------------------------------------------------------------------

# モデルの定義
visible_units = 28 * 28 +10  # Mnist画像サイズ
hidden_units = 4096  # 隠れ層のユニット数
# モデルの初期化
rbm = GBRBM(visible_units, hidden_units).to(device)

# パラメータの挿入
rbm.load_state_dict(ex_rbm.state_dict())
print('Complete parameter adaptation.')

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.SGD(rbm.parameters(), lr=0.01)

# 学習率の減衰スケジューラの定義
# step_sizeは学習率を減衰させるエポックの間隔を指定
# scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

# 学習
losses = []
num_epochs = 10
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# accuracyの値を格納するためのリスト
train_accuracies = []
zeros_val_accuracies = []
rand_val_accuracies = []
rbm.train()
for epoch in range(num_epochs):
    start_time = time.time()  # エポック開始時刻
    epoch_losses = []

    # tqdm を使用してプログレスバー表示
    train_epoch_accuracies = []
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        rbm.train()
        data = data.view(data.size(0), -1).to(device)  # データの平滑化
        v0 = data
        ph0 = rbm(v0)
        h0 = rbm.sample_hidden(ph0)

                # Contrastive Divergenceのサンプリングステップ数
        vk = v0  # 初期化
        for k in range(10):
            hk = rbm.sample_hidden(rbm(vk))
            vk = rbm.backward(hk)

        rbm.zero_grad()
        positive_phase = torch.matmul(v0.t(), ph0)
        negative_phase = torch.matmul(vk.t(), rbm(vk))
        loss = criterion(positive_phase - negative_phase, torch.zeros_like(positive_phase))
        # トレーニングデータのaccを計算
        train_predictions = torch.argmax(rbm.backward(rbm(data))[:, 28*28:], dim=1)  # 後半の10要素を取得
        train_ground_truth = torch.argmax(data[:, 28*28:], dim=1)
        # print(f'test {train_ground_truth.cpu().numpy()}')
        # print(f'prediction {train_predictions.cpu().numpy()}')
        train_accuracy = accuracy_score(train_ground_truth.cpu().numpy(), train_predictions.cpu().numpy())
        train_epoch_accuracies.append(train_accuracy)

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    # Trainデータの平均正確さを計算
    train_avg_accuracy = np.mean(train_epoch_accuracies)
    train_accuracies.append(train_avg_accuracy)

    # Validation の正確さを計算
    rbm.eval()
    zeros_val_epoch_accuracies = []
    rand_val_epoch_accuracies = []
    # zeors label の作成
    zeros_label = torch.zeros(batch_size, 10).to(device)
    # rand label の作成
    rand_label = torch.randn(batch_size, 10).to(device)

    for data in tqdm(train_val_loader, desc=f'Testing Epoch {epoch+1}/{num_epochs}', leave=False):
        bind_data = data.view(data.size(0), -1).to(device)
        img_data = bind_data[:, :28*28]
        zeros_data = torch.cat((img_data, zeros_label), dim=1)
        rand_data = torch.cat((img_data, rand_label), dim=1)
        zeros_val_predictions = torch.argmax(k_sampling(10, rbm, zeros_data)[:, 28*28:], dim=1)
        rand_val_predictions = torch.argmax(k_sampling(10, rbm, rand_data)[:, 28*28:], dim=1)
        val_ground_truth = torch.argmax(bind_data[:, 28*28:], dim=1)
        zeros_val_accuracy = accuracy_score(val_ground_truth.cpu().numpy(), zeros_val_predictions.cpu().numpy())
        rand_val_accuracy = accuracy_score(val_ground_truth.cpu().numpy(), rand_val_predictions.cpu().numpy())
        zeros_val_epoch_accuracies.append(zeros_val_accuracy)
        rand_val_epoch_accuracies.append(rand_val_accuracy)

    # Validation の平均正確さを計算
    zeros_val_avg_accuracy = np.mean(zeros_val_epoch_accuracies)
    zeros_val_accuracies.append(zeros_val_avg_accuracy)
    rand_val_avg_accuracy = np.mean(rand_val_epoch_accuracies)
    rand_val_accuracies.append(rand_val_avg_accuracy)

    end_time = time.time()  # エポック終了時刻
    elapsed_time = end_time - start_time  # エポック実行時間
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)
    # 学習経過の表示
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}, Time: {elapsed_time:.2f} seconds')
    # 訓練データとValidationのAccuracyを表示
    print(f'Train Accuracy: {train_avg_accuracy:.4f}, Zeros Val Accuracy: {zeros_val_avg_accuracy:.4f}, Rand Val Accuracy: {rand_val_avg_accuracy:.4f}')

    # 学習率の減衰
    # scheduler.step()

# フォルダを作成
os.makedirs('evaluation', exist_ok=True)
os.makedirs('evaluation/reconstructed_images', exist_ok=True)

# モデルの保存
model_path = './evaluation/model.pth'
torch.save(rbm.state_dict(), model_path)
print(f'Model parameters saved to {model_path}')

# 損失の変化グラフの表示
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./evaluation/TrainLoss.png')
plt.show()

# Train Accuracy・Val Accuracyをプロット
plt.figure()
plt.plot(train_accuracies, label='TrainAcc')
plt.plot(rand_val_accuracies, label='ZerosValAcc')
plt.plot(rand_val_accuracies, label='RandValAcc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./evaluation/Accuracy.png')
plt.show()

with torch.no_grad():
    num_samples_batch = 1  # 表示する総サンプル数
    num_samples_to_display = 10  # 表示する総サンプル数
    displayed_samples = 0

    for i, data in enumerate(train_val_loader):
        if displayed_samples >= num_samples_batch:
            break

        # ラベル部分を分離
        img_data = data[:, :28*28]

        # データも明示的にデバイスに移動させる
        data = data.view(data.size(0), -1).to(device)
        reconstructed_data = rbm.backward(rbm(data))
        reconstructed_data = reconstructed_data[:, :28*28]
        reconstructed_data = reconstructed_data.view(data.size(0), 28, 28)

        # 元画像と再構築画像の表示
        for j in range(data.size(0)):
            if displayed_samples >= num_samples_to_display:
                break

            original_image = img_data[j].view(28, 28).cpu().numpy()
            reconstructed_image = reconstructed_data[j].cpu().numpy()

            # ファイルのPATH
            reconstructed_path = f'./evaluation/reconstructed_images/reconstructed_{displayed_samples + 1}.png'
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image')
            plt.savefig(reconstructed_path)

            plt.close()

            displayed_samples += 1

        if displayed_samples >= num_samples_to_display:
            break