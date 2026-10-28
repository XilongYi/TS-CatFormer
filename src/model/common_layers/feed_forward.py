import torch
import torch.nn as nn


def init_layer(layer):
    for p in layer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            nn.init.constant_(p.data, 0.0)
    if isinstance(layer, nn.Embedding):
        layer.weight.data[0].zero_()

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        # self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

        self.init_param()

    def init_param(self):
        init_layer(self.layer1)
        init_layer(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.relu(x)
        x = self.layer2(x)
        return x
