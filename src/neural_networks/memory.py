import torch
import torch.nn as nn
from typing import Tuple
from timeit import default_timer as timer
T = torch.Tensor

class Normal_LSTMCell(nn.Module):
    """
    Implementation of a normal LSTM Cell.
    """
    def __init__(self, input_sz: int, hidden_sz: int):
        """
        Creates an LSTM cell that takes inputs of the given size and outputs a Tensor of the given hidden size.
        """
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        # Paramters for the input grouped by the 4 lstm gates
        self.weight_ih = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        # Paramters for the hidden state grouped by the 4 lstm gates
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        # The bias for the input for all 4 lstm gates
        self.bias_ih  = nn.Parameter(torch.Tensor(input_sz * 4))
        # The bias for the hidden state all 4 lstm gates
        self.bias_hh  = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: T, hidden_state: Tuple[T]=None) -> Tuple[T, Tuple[T, T]]:
        """
        Assumes x is of shape (batch, input_size)
        """
        if hidden_state is None:
            h, c = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h, c = hidden_state
         
        HS = self.hidden_size

        # batch the computations into a single matrix multiplication
        gates = (x @ self.weight_ih + self.bias_ih) + (h @ self.weight_hh + self.bias_hh)
        i, f, g, o = (
            torch.sigmoid(gates[:, :HS]), # input
            torch.sigmoid(gates[:, HS:HS*2]), # forget
            torch.tanh(gates[:, HS*2:HS*3]),
            torch.sigmoid(gates[:, HS*3:]), # output
        )
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class I_TO_I_LSTMCell(nn.Module):
    """
    Implementation of an LSTM Cell that allow multidimensional feature inputs
    """
    def __init__(self, input_dim: Tuple[int], ):
        """
        Creates an LSTM cell that takes inputs of the given dimensions and outputs a Tensor of the same dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        # Paramters for the input grouped by the 4 lstm gates
        self.weight_ih = nn.Parameter(torch.Tensor(4, 1, *input_dim))
        # Paramters for the hidden state grouped by the 4 lstm gates
        self.weight_hh = nn.Parameter(torch.Tensor(4, 1, *input_dim))
        # The bias for the input for all 4 lstm gates
        self.bias_ih = nn.Parameter(torch.Tensor(4, 1, *input_dim))
        # The bias for the hidden state all 4 lstm gates
        self.bias_hh  = nn.Parameter(torch.Tensor(4, 1, *input_dim))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: T, hidden_state: Tuple[T]=None) -> Tuple[T, T]:
        """
        Assumes x is of shape (batch, *input_size)
        """
        if hidden_state is None:
            h, c = (torch.zeros(*self.input_dim).to(x.device), 
                        torch.zeros(*self.input_dim).to(x.device))
        else:
            h, c = hidden_state

        # batch the computations into a single matrix multiplication
        gates = (self.weight_ih * x + self.bias_ih) + (self.weight_hh * h + self.bias_hh)
        i, f, g, o = (
            torch.sigmoid(gates[0]), # input
            torch.sigmoid(gates[1]), # forget
            torch.tanh(gates[2]),
            torch.sigmoid(gates[3]), # output
        )
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class Multi_dim_LSTMCell(nn.Module):
    """
    Implementation of an LSTM Cell that allow multidimensional feature inputs
    """
    def __init__(self, input_dim: Tuple[int], hidden_dim: Tuple[int]):
        """
        Creates an LSTM cell that takes inputs of the given dimensions and outputs a Tensor of the same dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Paramters for the input grouped by the 4 lstm gates
        self.weight_ih = nn.Parameter(torch.Tensor(4, *input_dim, *hidden_dim))
        # Paramters for the hidden state grouped by the 4 lstm gates
        self.weight_hh = nn.Parameter(torch.Tensor(4, *input_dim, *hidden_dim))
        # The bias for the input for all 4 lstm gates
        self.bias_ih = nn.Parameter(torch.Tensor(4, *input_dim))
        # The bias for the hidden state all 4 lstm gates
        self.bias_hh  = nn.Parameter(torch.Tensor(4, *input_dim))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: T, hidden_state: Tuple[T]=None) -> Tuple[T, T]:
        """
        Assumes x is of shape (batch, *input_size)
        """
        if hidden_state is None:
            h, c = (torch.zeros(*self.hidden_dim).to(x.device), 
                        torch.zeros(*self.hidden_dim).to(x.device))
        else:
            h, c = hidden_state

        wi =torch.tensordot(x, self.weight_ih, ([1,2], [1,2]))
        wh = torch.tensordot(h, self.weight_hh, ([0,1], [1,2]))
        # batch the computations into a single matrix multiplication
        gates = (wi + self.bias_ih) + (wh + self.bias_hh)
        i, f, g, o = (
            torch.sigmoid(gates[:,0]), # input
            torch.sigmoid(gates[:,1]), # forget
            torch.tanh(gates[:,2]),
            torch.sigmoid(gates[:,3]), # output
        )
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


def printNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    print(num_params)


if __name__ == "__main__":
    inp = torch.randn(32,100,91)

    # Data preperation
    preperation = timer()
    s = tuple(inp.size())
    flat_inp = inp.flatten(1)
    preperation = timer() - preperation


    py_lstm = Normal_LSTMCell(flat_inp.size(1), flat_inp.size(1))
    gru = torch.nn.GRUCell(flat_inp.size(1), flat_inp.size(1))
    i_to_i_lstm = I_TO_I_LSTMCell(tuple(inp[0].size()))
    multi_dim_lstm = Multi_dim_LSTMCell(tuple(inp.shape[1:]), tuple(inp.shape[1:]))

    t1 = timer()
    o1, h1 = py_lstm(flat_inp)
    o1 = o1.view(*s)
    t1 = timer() - t1 + preperation

    t2 = timer()
    o2 = gru(flat_inp)
    o2 = o2.view(*s)
    t2 = timer() - t2 + preperation

    t3 = timer()
    o3, h3 = i_to_i_lstm(inp)
    t3 = timer() - t3

    t4 = timer()
    o4, h4 = multi_dim_lstm(inp)
    o4 = o4.view(*s)
    t4 = timer() - t4 + preperation

    printNumberParams(py_lstm)
    printNumberParams(gru)
    printNumberParams(i_to_i_lstm)
    printNumberParams(multi_dim_lstm)


    print(o1.size())
    print(o2.size())
    print(o3.size())
    print(o4.size())
    print(t1, t2, t3, t4)