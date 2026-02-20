import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
            nn.Conv1d(oc, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
        )
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., :x.size(2)]
        return y + self.res(x)

class TCN(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        chans = [192, 192, 192, 192]
        layers = []

        for i, c in enumerate(chans):
            layers.append(
                TemporalBlock(
                    feature_dim if i == 0 else chans[i-1],
                    c, 2 ** i
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(chans[-1], num_classes)

    def masked_pool(self, x, m):
        m = m.unsqueeze(1)
        return (x * m).sum(2) / (m.sum(2) + 1e-6)

    def forward(self, x, m):
        x = self.tcn(x)
        x = self.masked_pool(x, m)
        return self.fc(x)