import torch
from torch import nn

class CGAN_Discriminator(nn.Module):
    """
    CGAN discriminator model

    input shape: number of total columns (target features+condition) in multivariate data
    hidden units: number of neurons in the layers
    output shape: 1
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        # input shape is number of columns
        # output shape is 1 (for probability of 0 or 1)

        # pass concatenated target+conditions through nn.Sequential
        self.main = nn.Sequential(
            # input is INPUT SHAPE // number of columns/channels
            nn.Conv1d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=3,
                         stride=2,
                         padding=1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
        )

        self.linearize = nn.Sequential(
            nn.Linear(in_features=hidden_units*3,
                      out_features=output_shape),
            nn.Sigmoid() # output of probability
        )

    def forward(self, target, conditions):
        """
        takes target and conditions as batches
        activation function: sigmoid
        """
        # concatenating and transposing
        data = torch.cat([target, conditions], axis=2)
        data_in = torch.transpose(data, 1, 2)

        # forward pass through nn.Main
        out = self.main(data_in)
        # forward pass through nn.linear + nn.sigmoid
        out = self.linearize(out)

        return out


class CGAN_Generator(nn.Module):
    """
    CGAN generator model

    input shape: number of total columns (target features+condition) in multivariate data
    hidden units: number of neurons in the layers
    output shape: number of feature columns (total columns-condition) in multivariate data
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        # input shape is number of columns
        # output shape is 2 (for probability of 0 or 1)

        # pass concatenated target+conditions through nn.Sequential
        self.main = nn.Sequential(
            # input is INPUT SHAPE // number of columns/channels
            nn.ConvTranspose1d(in_channels=input_shape,
                               out_channels=hidden_units,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=hidden_units,
                               out_channels=hidden_units,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=45,
                      kernel_size=3,
                      stride=1,
                      padding=1)
    	)

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise, conditions):
        """
        takes target and conditions as batches
        activation function: sigmoid
        """
        # concatenating and transposing
        data = torch.cat([noise, conditions], axis=2)
        data_in = torch.transpose(data, 1, 2)

        # forward pass through nn.Main
        out = self.main(data_in)

        # tranposing the data for activation function
        data_in = torch.transpose(out, 1, 2)

        # forward pass through nn.Activation
        out = self.sigmoid(data_in)

        return out

