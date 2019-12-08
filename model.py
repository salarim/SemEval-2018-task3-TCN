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



class LSTM_classifier(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(LSTM_classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print(input.shape)
        output,(_, _) = self.lstm(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y

class LSTM_classifier_bidirectional(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(LSTM_classifier_bidirectional, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional = True)
        self.decoder = nn.Linear(2  * hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print(input.shape)
        output,(_, _) = self.lstm(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y


class GRU_classifier(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(GRU_classifier, self).__init__()
        self.GRU = nn.GRU(input_size, hidden_size, batch_first = True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print(input.shape)
        output,_ = self.GRU(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y

class GRU_classifier_bidirectional(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(GRU_classifier_bidirectional, self).__init__()
        self.GRU = nn.GRU(input_size, hidden_size, batch_first = True, bidirectional = True)
        self.decoder = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print(input.shape)
        output,_ = self.GRU(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y

class GRU_classifier_mlayers(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(GRU_classifier_mlayers, self).__init__()
        self.GRU = nn.GRU(input_size, hidden_size, batch_first = True, num_layers = num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        #print(input.shape)
        output,_ = self.GRU(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y

class RNN_classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN_classifier, self).__init__()
        self.RNN = nn.RNN(input_size, hidden_size, batch_first = True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #input = input.permute(1, 0, 2)
        #print(input.shape)
        output, _ = self.RNN(input)
        y = self.decoder(output[:,-1])
        y = self.sigmoid(y)
        return y

