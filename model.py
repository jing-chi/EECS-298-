import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, inputsize, hiddensize, num_layers, outputsize):
        super(LSTM, self).__init__()
        self.hiddensize = hiddensize
        self.num_layers = num_layers

        self.lstm = nn.LSTM(inputsize, hiddensize, num_layers, batch_first=True)
        self.fc = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hiddensize).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hiddensize).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out