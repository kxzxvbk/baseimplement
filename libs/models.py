import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, voc_size, embedding_dim=256, filters=None, out_channels=32):
        super(ConvNet, self).__init__()
        if filters is None:
            filters = [1, 2, 4, 8]
        self.embedding = torch.nn.Embedding(voc_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, i) for i in filters])
        self.y1_length = out_channels * len(filters)
        self.linear1 = nn.Linear(self.y1_length, 32)

    def forward(self, x):
        # print('Before embedding: ' + str(x.shape))
        x = self.embedding(x).unsqueeze(0).float()
        # print('After embedding: ' + str(x.shape))
        y1 = []
        for conv in self.convs:
            y1.append(conv(x).squeeze(0).max(1)[0].max(1)[0])
        y1 = torch.cat(y1, dim=0)
        y = torch.tanh(self.linear1(y1))
        return y


class TopModuleCNN(nn.Module):
    def __init__(self, voc_size, output_channel):
        super(TopModuleCNN, self).__init__()
        self.conv_net1 = ConvNet(voc_size, output_channel)
        self.conv_net2 = ConvNet(voc_size, output_channel)

    def forward(self, x, y):
        x1 = self.conv_net1(x)
        y1 = self.conv_net2(y)

        return torch.sigmoid(x1.dot(y1.t())).unsqueeze(0)
