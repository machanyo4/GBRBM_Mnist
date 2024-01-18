import torch
import torch.nn as nn

# GPUが利用可能か確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gaussian Bernoulli RBMの定義
class GBRBM(nn.Module):
    """ Gaussian-Bernoulli Restricted Boltzmann Machines (GBRBM) """
    def __init__(self, visible_units, hidden_units):
        super(GBRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units).to(device))
        self.visible_bias = nn.Parameter(torch.zeros(visible_units).to(device))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_units).to(device))

    def forward(self, v):
        h = torch.sigmoid(torch.matmul(v, self.W) + self.hidden_bias)
        return h

    def sample_hidden(self, h):
        return torch.bernoulli(h)

    def backward(self, h):
        v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.visible_bias)
        return v

    def sample_visible(self, v):
        return torch.normal(v, torch.ones_like(v)).to(device)