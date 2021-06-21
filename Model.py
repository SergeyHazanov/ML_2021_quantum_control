import torch.nn as nn

NETWORK_INPUT_SIZE = 2
NETWORK_OUTPUT_SIZE = 6
LAYER_SIZES = [80] * 3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.layer1 = nn.Linear(NETWORK_INPUT_SIZE, LAYER_SIZES[0])

        for hidden_i in range(len(LAYER_SIZES) - 1):
            self.hidden_layers.append(nn.Linear(LAYER_SIZES[hidden_i], LAYER_SIZES[hidden_i + 1]))

        self.layer2 = nn.Linear(LAYER_SIZES[-1], NETWORK_OUTPUT_SIZE)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)

        for layer in self.hidden_layers:
            out = layer(out)
            out = self.activation(out)

        out = self.layer2(out)
        return out
