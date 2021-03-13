import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple MLP for classification
    """

    def __init__(self, input_dim, out_dim, layers, drop_rate):
        super(MLP, self).__init__()
        print(f"MLP architecture: {layers}")
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, layers[0])
        self.dropout1 = nn.Dropout(drop_rate)
        self.layer2 = nn.Linear(layers[0], layers[1])
        self.layer3 = nn.Linear(layers[1], layers[2])
        self.layer_out = nn.Linear(layers[2], out_dim)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer_out(x)
        return x
