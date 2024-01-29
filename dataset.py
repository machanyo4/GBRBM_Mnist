import torch

class OnlyMNISTImgDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        # 画像データのみ使用
        data, _ = self.original_dataset[index]

        # 画像データを平滑化
        data = data.view(-1)

        return data

class ExtendedMNISTDatasetOnehot(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]

        # 画像データを平滑化
        data = data.view(-1)

        # ラベルをワンホットベクトルに変換
        one_hot_label = torch.zeros(10)  # ラベルのクラス数に合わせて適切な次元数に変更
        one_hot_label[label] = 1

        # 末尾にワンホットベクトルを結合
        extended_data = torch.cat((data, one_hot_label))

        return extended_data

class ExtendedMNISTDatasetZeros(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]

        # 画像データを平滑化
        data = data.view(-1)

        # ラベルをZeroベクトルに変換
        zero_label = torch.zeros(10)  # ラベルのクラス数に合わせて適切な次元数に変更
        # 末尾にZeroベクトルを結合
        extended_data = torch.cat((data, zero_label))

        return extended_data

class ExtendedMNISTDatasetZeros(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        data, label = self.original_dataset[index]

        # 画像データを平滑化
        data = data.view(-1)

        # ラベルをZeroベクトルに変換
        zero_label = torch.zeros(10)  # ラベルのクラス数に合わせて適切な次元数に変更
        # 末尾にZeroベクトルを結合
        extended_data = torch.cat((data, zero_label))

        return extended_data
    
# k-sampling
def k_sampling(k, model, data):
    for i in range(k):
        data = model.backward(model(data))
    return data