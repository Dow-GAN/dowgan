import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Time_GAN_module(nn.Module):
    """
    Class from which a module of the Time GAN Architecture can be constructed, 
    consisting of a n_layer stacked RNN layers and a fully connected layer
    
    input_size = dim of data (depending if module operates on latent or non-latent space)
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers, activation=torch.sigmoid, rnn_type="gru"):
        super(Time_GAN_module, self).__init__()

        # Parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sigma = activation
        self.rnn_type = rnn_type

        #Defining the layers
        # RNN Layer
        if self.rnn_type == "gru":
          self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        elif self.rnn_type == "rnn":
          self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first = True) 
        elif self.rnn_type == "lstm": # input params still the same for lstm
          self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first = True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
    
            batch_size = x.size(0)

            # Initializing hidden state for first input using method defined below
            if self.rnn_type in ["rnn", "gru"]:
              hidden = self.init_hidden(batch_size)
            elif self.rnn_type == "lstm": # additional initial cell state for lstm
              h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).float()
              c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).float()
              hidden = (h0, c0)
            # Passing in the input and hidden state into the model and obtaining outputs
            out, hidden = self.rnn(x, hidden)
        
            # Reshaping the outputs such that it can be fit into the fully connected layer
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
            
            if self.sigma == nn.Identity:
                idendity = nn.Identity()
                return idendity(out)
                
            out = self.sigma(out)
            
            # HIDDEN STATES WERDEN IN DER PAPER IMPLEMENTIERUNG AUCH COMPUTED, ALLERDINGS NICHT BENUTZT?
            
            return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden