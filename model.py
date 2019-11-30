import torch
from torch import nn
import sys
from tcn import TemporalConvNet


class TCN(nn.Module):

    def __init__(self, input_size
                 , output_size, num_channels,
                 kernel_size=2, dropout=0.3):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        y = self.tcn(input.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y[:,-1,:])
        y = self.sigmoid(y)
        return y

