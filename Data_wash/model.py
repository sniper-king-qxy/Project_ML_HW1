import torch
import torch.nn as nn
class Config_data:
    input_size = 3
    hidden_size = 64
    output_size = 1
    num_layers = 3
    dropout_prob = 0.1


class FNN(nn.Module):
    def __init__(self, config):
        super(FNN, self).__init__()
        layers = []
        input_dim = config.input_size

        for _ in range(config.num_layers - 1):
            layers += [
                nn.Linear(input_dim, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob)
            ]
            input_dim = config.hidden_size

        layers.append(nn.Linear(input_dim, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)