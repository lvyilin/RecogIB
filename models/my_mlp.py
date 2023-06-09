import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, num_hiddens=6, num_dims=1024):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_hiddens):
            layers.append(nn.Linear(in_features, num_dims))
            layers.append(nn.LeakyReLU(inplace=True))
            in_features = num_dims
        layers.append(nn.Linear(in_features, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze()
