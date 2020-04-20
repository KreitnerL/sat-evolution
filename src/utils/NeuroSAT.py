import torch
import torch.nn as nn
import torch.optim as optim
T = torch.Tensor
embedding_size=128

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params

class NeuroSAT(nn.Module):
    """
    Simplified implementation of the message passing used in NeuroSAT https://arxiv.org/abs/1806.01261.
    """
    def __init__(self):
        super().__init__()
        hidden_message_layer = 3
        self.L_msg = nn.Sequential()
        self.C_msg = nn.Sequential()
        for i in range(hidden_message_layer):
            self.L_msg.add_module("Linear"+str(i),nn.Linear(embedding_size, embedding_size))
            self.C_msg.add_module("Linear"+str(i),nn.Linear(embedding_size, embedding_size))
            self.L_msg.add_module("ReLU"+str(i),nn.ReLU(inplace=True))
            self.C_msg.add_module("ReLU"+str(i),nn.ReLU(inplace=True))

        self.L_u = nn.GRUCell(embedding_size, embedding_size)
        self.C_u = nn.GRUCell(embedding_size, embedding_size)

        print(getNumberParams(self))

    def forward(self, l_t, c_t, A):
        # Update clause with messages from neighboring literals
        m_l = self.L_msg(l_t) 
        c_t_new = self.C_u(A.t() @ m_l, c_t)

        # Update literals with messages from neighboring clauses and their complements
        m_c = self.C_msg(c_t_new)
        l_t_new = self.L_u(A @ m_c, l_t)
        return l_t_new, c_t_new

if __name__ == "__main__":
    neuroSAT = NeuroSAT()
    l = T(20, embedding_size)
    c = T(91, embedding_size)
    A = T(20,91)
    for i in range(4):
        l , c = neuroSAT(l, c, A)
    print(l.shape, c.shape)