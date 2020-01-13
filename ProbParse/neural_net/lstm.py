import torch
import torch.nn as nn
import torch.nn.functional as F

from const import MAX_NAME_LENGTH, DEVICE


class Encoder(nn.Module):
    """
    Takes in an one-hot tensor of names and produces hidden state and cell state
    for decoder LSTM to use.

    input_size: Number of permitted letters
    hidden_size: Size of the hidden dimension
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialize LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step.

        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        """
        # input = input.view(1, self.batch_size, -1)
        lstm_out, hidden = self.lstm(input, hidden)
        return lstm_out, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))


class Decoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.
    Uses attention during forward phase

    input_size: Number of permitted letters
    hidden_size: Size of the hidden dimension
    output_size: Number of permitted letters
    num_layers: The number of layers in each LSTM nodule
    max_length: The max input length, characters are padded to meet this
    temp: Temperature for softmax
    """

    def __init__(self, input_size, hidden_size, output_size, max_length, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

        # Attention components (self.attn must have the output size of encoder's max name length)
        self.attn = nn.Linear(self.input_size + 2 * self.hidden_size, MAX_NAME_LENGTH)
        self.attn_combine = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs, mask):
        """
        Run LSTM through 1 time step

        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        - mask : 1 x batch_size x max length
        """
        
        # Incorporate attention to LSTM input
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=2)
        attn_out = self.attn(torch.cat((input, hidden_cat), dim=2))
        # Apply mask to attention, and set pads to negative infinity
        attn_out = torch.where(mask, attn_out, torch.tensor(float('-inf')).to(DEVICE))
        # Softmax the attention out
        attn_weights = F.softmax(attn_out, dim=2)
        attn_applied = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs.transpose(0, 1)).transpose(0, 1)
        attn_output = torch.cat((input, attn_applied), 2)
        attn_output = F.relu(self.attn_combine(attn_output))

        # Run LSTM
        lstm_out, hidden = self.lstm(attn_output, hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))


class PreTrainedDecoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.
    input_size: Number of permitted letters
    hidden_size: Size of the hidden dimension
    output_size: Number of permitted letters
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PreTrainedDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize LSTM - Notice it does not accept output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step
        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        """
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))
