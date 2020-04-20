import torch
import torch.nn as nn
from neural_networks.feature_collection import Feature_Collection
from neural_networks.pool_conv_sum_nonlin_pool import Pool_conv_sum_nonlin_pool, get_full_shape
from collections import Counter
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
    Updates the literal and clause embeddings by using a messaging system that combines the approach from https://arxiv.org/abs/1903.04671 and https://arxiv.org/abs/1711.08028.
    Consider the SAT problem as a bidirectional graph where each literal (polarity matters) and each clause is a node. There is an edge between literal l and clause c if c inherits l.
    Every node maintains an embedding of size d that is iteratively refined at each time step. Every step, all clauses receive a messsage of size d from their neighboring literals 
    and update their embeddings. 
    Then, all literals receives a message of size d from their neighboring clauses, as well as the embedding of their complement and update their embedding accordingly. The network learns
    which messages to send and how to precess them.
    """
    def __init__(
        self, 
        adjacency_matrix: T,
        literal_embedding_key: tuple,
        clause_embedding_key: tuple,
        literal_embedding_size,
        clause_embedding_size,
        global_embedding_size,
        additional_information_size: dict = dict(),
        activation_func: type(F.relu) = F.relu,
        global_pool_func: type(torchMax) = torchMax):
        """
        :param adjacency_matrix: adjacency matrix between literals and clauses. Tensor of size Ex2G where E = #clauses, G = #variables, respectively
        :param embedding_size: the size of the embedding for the literal and clause tensors, i.e. the number of channels
        :param additional_information_size: dictionary storing the code and number of channels for all additional features that are used by the update gates
        :literal_embedding_key: code of the literal_embedding
        :clause_embedding_key: code of the clause_embedding
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        self.A = adjacency_matrix
        self.A_t = adjacency_matrix.t()
        self.literal_embedding_key = literal_embedding_key
        self.clause_embedding_key = clause_embedding_key
        self.global_embedding_key = tuple([1]+[0]*(len(literal_embedding_key)-1))

        # Literal message layer
        self.L_msg = conv_map[sum(literal_embedding_key)](literal_embedding_size, clause_embedding_size, 1)
        # Clause message layer
        self.C_msg = conv_map[sum(clause_embedding_key)](clause_embedding_size, literal_embedding_size, 1)

        # Literal update layer
        self.L_u = Pool_conv_sum_nonlin_pool(
            num_input_channels = Counter({literal_embedding_key: 3*literal_embedding_size}) + Counter(additional_information_size),
            num_output_channels = literal_embedding_size,
            output_stream_codes = [literal_embedding_key]
        )
        # Clause update layer
        self.C_u = Pool_conv_sum_nonlin_pool(
            num_input_channels = Counter({clause_embedding_key: 2*clause_embedding_size}) + Counter(additional_information_size),
            num_output_channels = clause_embedding_size,
            output_stream_codes = [clause_embedding_key]
        )

        self.G_u = Pool_conv_sum_nonlin_pool(
            num_input_channels = Counter({self.global_embedding_key: literal_embedding_size + clause_embedding_size + global_embedding_size}) + Counter(additional_information_size),
            num_output_channels = global_embedding_size,
            output_stream_codes = [self.global_embedding_key]
        )

    def forward(self, L_t, C_t, U_t, additional_information: Feature_Collection = None):
        """
        :param l_t: 
        """
        # Providing "static" additional information every time helps the network to focus solely on the messages instead of trying to remember the input.
        C_shape = get_full_shape(self.clause_embedding_key, C_t.shape)
        L_shape = get_full_shape(self.literal_embedding_key, L_t.shape)
        C_U_shape = get_full_shape(self.global_embedding_key, C_t.shape)
        L_U_shape = get_full_shape(self.global_embedding_key, L_t.shape)
        U_shape = get_full_shape(self.global_embedding_key, U_t.shape)

        # Update clauses with messages from neighboring literals
        M_l = self.L_msg(L_t) @ self.A_t
       # Update clauses with messages from neighboring literals
        C_t_new = self.C_u(Feature_Collection([C_t.view(C_shape), M_l.view(C_shape)]).addAll(additional_information), pool=False)

        # Update literals with messages from neighboring clauses and their complements
        M_c = self.C_msg(C_t_new) @ self.A
        L_t_new = self.L_u(Feature_Collection([L_t.view(L_shape), M_c.view(L_shape), flip(L_t, 2).view(L_shape)]).addAll(additional_information), pool=False)

        U_t_new = self.G_u(Feature_Collection([L_t_new.sum(-1).view(L_U_shape), C_t_new.sum(-1).view(C_U_shape), U_t.view(U_shape)]).addAll(additional_information), pool=False)
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
    test = Message_Passing_Core(
        adjacency_matrix = T(E,2*G), 
        literal_embedding_key=(1,0,0,1), 
        clause_embedding_key=(1,1,0,0),
        literal_embedding_size=literal_embedding_size,
        clause_embedding_size=clause_embedding_size,
        global_embedding_size=global_embedding_size)
    l_t = T(1,literal_embedding_size, 10, 2*G)
    c_t = T(1,clause_embedding_size, 10, E)
    u_t = T(1,global_embedding_size,10)
    l_t_new, c_t_new, u_t_new = test(l_t, c_t, u_t, None)
    print(l_t_new.shape, c_t_new.shape, u_t_new.shape)