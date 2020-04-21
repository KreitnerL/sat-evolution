import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
T = torch.Tensor
torchMax = lambda *x: T.max(*x)[0]

conv_map = {
    0: nn.Conv1d,
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

class Message_Passing_Core(nn.Module):
    """
    Updates the literal and clause embeddings by using a messaging system as described by https://arxiv.org/abs/1903.04671.
    Consider the SAT problem as a bidirectional graph where each literal (polarity matters) and each clause is a node. There is an edge between literal l and clause c if c inherits l.
    Every node maintains an embedding of that is iteratively refined at each time step. Every step, all clauses receive a messsage of size d_c from their neighboring literals 
    and update their embeddings. 
    Then, all literals receives a message of size d_l from their neighboring clauses, as well as the embedding of their complement and update their embedding accordingly. 
    Finally the global embedding is refined by takiing all clause and literal embeddings into account.
    The network learns which messages to send and how to precess them.
    """
    def __init__( self, literal_embedding_size, clause_embedding_size, global_embedding_size):
        """
        :param literal_embedding_size: size of the literal node embeddings
        :param clause_embedding_size: size of the clause node embeddings
        :param global_embedding_size: size of the global embedding
        """
        super().__init__()
        # Literal message layer
        self.L_msg = nn.Conv1d(literal_embedding_size, clause_embedding_size, 1)
        # Clause message layer
        self.C_msg = nn.Conv1d(clause_embedding_size, literal_embedding_size, 1)

        # Literal update layer
        self.L_u = nn.Sequential(nn.Conv1d(3*literal_embedding_size, literal_embedding_size, 1), nn.LeakyReLU(inplace=True))
        # Clause update layer
        self.C_u = nn.Sequential(nn.Conv1d(2*clause_embedding_size, clause_embedding_size, 1), nn.LeakyReLU(inplace=True))
        # Global update layer
        self.U_u = nn.Sequential(nn.Conv1d(literal_embedding_size + clause_embedding_size + global_embedding_size, global_embedding_size, 1), nn.LeakyReLU(inplace=True))

    def forward(self, L_t, C_t, U_t, A, A_t):
        """
        :param L_t: Literal embeddings
        :param C_t: Clause embeddings
        :param U_t: Global embedding
        :param A: Adjacency matrix
        :param A_t: Adjacency matrix transponed
        """

        # Update clauses with messages from neighboring literals
        M_l = self.L_msg(L_t) @ self.A_t
       # Update clauses with messages from neighboring literals
        C_t_new = self.C_u(torch.cat([C_t, M_l],1))

        # Update literals with messages from neighboring clauses and their complements
        M_c = self.C_msg(C_t_new) @ self.A
        L_t_new = self.L_u(torch.cat([L_t, M_c, flip(L_t, 2)],1))

        # Update global embedding with embeddings from clauses and literals
        U_t_new = self.U_u(torch.cat([L_t_new.sum(-1, keepdim=True), C_t_new.sum(-1, keepdim=True), U_t],1))
        return L_t_new, C_t_new, U_t_new

def flip(A: T, dim):
    """
    Swaps the 1st half of the rows in dimension dim of a matrix with the 2nd half.
    :param dim: dimension on which to perform the flip. Length of this dim must be a multiple of 2
    """
    s = A.shape
    return A.view(*s[:dim],2,int(s[dim]/2),*s[dim+1:]).flip(dim).view(*s[:dim], s[dim], *s[dim+1:])

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params


if __name__ == "__main__":
    pass
    # Usage example. Note that you cannot run this in this in this file, because the dependencies will not be loaded correctly.
    G = 20
    E = 91
    literal_embedding_size = 80
    clause_embedding_size = 60
    global_embedding_size = 5
    test = Message_Passing_Core(literal_embedding_size, clause_embedding_size, global_embedding_size)
    l_t = T(1,literal_embedding_size, 10, 2*G)
    c_t = T(1,clause_embedding_size, 10, E)
    u_t = T(1,global_embedding_size,10)
    A = T(E,2*G)
    l_t_new, c_t_new, u_t_new = test(l_t, c_t, u_t, A)
    print(l_t_new.shape, c_t_new.shape, u_t_new.shape)