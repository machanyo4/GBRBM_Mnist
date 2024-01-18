from model import GBRBM
from dataset import ExtendedMNISTDatasetOnehot
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

# GPUが利用可能か確認
print("GPU available : ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの変換
transform = transforms.Compose([transforms.ToTensor()])

# MNISTデータセットのダウンロード
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 拡張データセットの作成
train_dataset = ExtendedMNISTDatasetOnehot(train_dataset)
test_dataset = ExtendedMNISTDatasetOnehot(test_dataset)

# データローダーの作成
batch_size = 512
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

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
test_accuracies = []

for epoch in range(num_epochs):
    start_time = time.time()  # エポック開始時刻
    epoch_losses = []

    # tqdm を使用してプログレスバー表示
    for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        data = data.view(data.size(0), -1).to(device)  # データの平滑化
        v0 = data
        ph0 = rbm(v0)
        h0 = rbm.sample_hidden(ph0)

        # Contrastive Divergenceのサンプリングステップ数
        for k in range(10):
            vk = rbm.backward(h0)
            pvk = rbm(vk)
            hk = rbm.sample_hidden(pvk)

        rbm.zero_grad()
        positive_phase = torch.matmul(v0.t(), ph0)
        negative_phase = torch.matmul(vk.t(), pvk)
        loss = criterion(positive_phase - negative_phase, torch.zeros_like(positive_phase))
        # トレーニングデータのaccを計算
        train_predictions = torch.argmax(rbm(data)[:, 28*28:], dim=1)  # 後半の10要素を取得
        train_ground_truth = torch.argmax(data[:, 28*28:], dim=1)
        train_accuracy = accuracy_score(train_ground_truth.cpu().numpy(), train_predictions.cpu().numpy())
        train_accuracies.append(train_accuracy)

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    # テストデータの正確さを計算
    test_epoch_accuracies = []
    for data in tqdm(test_loader, desc=f'Testing Epoch {epoch+1}/{num_epochs}', leave=False):
        data = data.view(data.size(0), -1).to(device)
        test_predictions = torch.argmax(rbm(data)[:, 28*28:], dim=1)
        test_ground_truth = torch.argmax(data[:, 28*28:], dim=1)
        test_accuracy = accuracy_score(test_ground_truth.cpu().numpy(), test_predictions.cpu().numpy())
        test_epoch_accuracies.append(test_accuracy)

    # テストデータの平均正確さを計算
    test_avg_accuracy = np.mean(test_epoch_accuracies)
    test_accuracies.append(test_avg_accuracy)

    end_time = time.time()  # エポック終了時刻
    elapsed_time = end_time - start_time  # エポック実行時間
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)
    # 学習経過の表示
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}, Time: {elapsed_time:.2f} seconds')
    # 訓練データとテストデータのAccuracyを表示
    print(f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_avg_accuracy:.4f}')

    # 学習率の減衰
    # scheduler.step()

# フォルダを作成
os.makedirs('evaluation', exist_ok=True)
os.makedirs('evaluation/original_images', exist_ok=True)
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
# plt.show()
plt.savefig('./evaluation/TrainLoss.png')

# Train Accuracyをプロット
plt.plot(train_accuracies, label='TrainAcc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./evaluation/TrainAcc.png')

# Test Accuracyをプロット
plt.plot(test_accuracies, label='TestAcc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('./evaluation/TestAcc.png')

with torch.no_grad():
    num_samples_batch = 1  # 表示する総サンプル数
    num_samples_to_display = 10  # 表示する総サンプル数
    displayed_samples = 0

    for i, data in enumerate(test_loader):
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
            original_path = f'./evaluation/original_images/original_{displayed_samples + 1}.png'
            reconstructed_path = f'./evaluation/reconstructed_images/reconstructed_{displayed_samples + 1}.png'
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            plt.savefig(original_path)
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image')
            plt.savefig(reconstructed_path)

            plt.close()

            displayed_samples += 1

        if displayed_samples >= num_samples_to_display:
            break