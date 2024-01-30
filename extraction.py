from model import GBRBM
from dataset import OnlyMNISTImgDataset
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

# GPUが利用可能か確認
print("GPU available : ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの変換
transform = transforms.Compose([transforms.ToTensor()])

# MNISTデータセットのダウンロード
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 拡張データセットの作成
train_dataset = OnlyMNISTImgDataset(train_dataset)
test_dataset = OnlyMNISTImgDataset(test_dataset)

# データローダーの作成
batch_size = 512
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

# モデルの定義
visible_units = 28 * 28  # Mnist画像サイズ
hidden_units = 4096  # 隠れ層のユニット数
rbm = GBRBM(visible_units, hidden_units).to(device)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.SGD(rbm.parameters(), lr=0.01)

# 学習率の減衰スケジューラの定義
# step_sizeは学習率を減衰させるエポックの間隔を指定
# scheduler = StepLR(optimizer, step_size=500, gamma=0.2)

# 学習
losses = []
num_epochs = 300

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
        vk = v0  # 初期化
        for k in range(10):
            hk = rbm.sample_hidden(rbm(vk))
            vk = rbm.backward(hk)

        rbm.zero_grad()
        positive_phase = torch.matmul(v0.t(), ph0)
        negative_phase = torch.matmul(vk.t(), rbm(vk))
        loss = criterion(positive_phase - negative_phase, torch.zeros_like(positive_phase))
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    end_time = time.time()  # エポック終了時刻
    elapsed_time = end_time - start_time  # エポック実行時間
    mean_loss = np.mean(epoch_losses)
    losses.append(mean_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}, Time: {elapsed_time:.2f} seconds')

    # 学習率の減衰
    # scheduler.step()

# フォルダを作成
os.makedirs('extraction', exist_ok=True)
os.makedirs('extraction/reconstructed_images', exist_ok=True)

# モデルの保存
model_path = './extraction/model.pth'
torch.save(rbm.state_dict(), model_path)
print(f'Model parameters saved to {model_path}')

# 損失の変化グラフの表示
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('./extraction/TrainLoss.png')


with torch.no_grad():
    num_samples_batch = 1  # 表示する総サンプル数
    num_samples_to_display = 10  # 表示する総サンプル数
    displayed_samples = 0

    for i, data in enumerate(test_loader):
        if displayed_samples >= num_samples_batch:
            break

        # ラベル部分を分離
        img_data = data

        # データも明示的にデバイスに移動させる
        data = data.view(data.size(0), -1).to(device)
        reconstructed_data = rbm.backward(rbm(data))
        reconstructed_data = reconstructed_data
        reconstructed_data = reconstructed_data.view(data.size(0), 28, 28)

        # 元画像と再構築画像の表示
        for j in range(data.size(0)):
            if displayed_samples >= num_samples_to_display:
                break

            original_image = img_data[j].view(28, 28).cpu().numpy()
            reconstructed_image = reconstructed_data[j].cpu().numpy()

            # ファイルに保存
            reconstructed_path = f'./extraction/reconstructed_images/reconstructed_{displayed_samples + 1}.png'
            
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