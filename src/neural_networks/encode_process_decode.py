import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from neural_networks.feature_collection import Feature_Collection
from neural_networks.pool_conv_sum_nonlin_pool import Pool_conv_sum_nonlin_pool, get_full_shape
from neural_networks.utils import init_weights
from neural_networks.message_passing_core import Message_Passing_Core
from collections import Counter
from typing import Tuple, List
from solvers.encoding import ProblemInstanceEncoding
NUM_DIMENSIONS = ProblemInstanceEncoding.NUM_DIMENSIONS
T = torch.Tensor
torchMax = lambda *x: torch.max(*x)[0]
conv_map = {
    0: nn.Conv1d,
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

class Encode_Process_Decode(nn.Module):
    """
    This class implements the encode-process-decode architecture as specified here https://arxiv.org/abs/1806.01261.
    Given a population of graph encoded SAT problem, it outputs appropiate action distributions per individual as well as a critic value (combined actor-critic model).
    """
    def __init__(
        self,
        embeddings: List[tuple],
        additional_information: dict = dict()):
        """
        :param embeddings: List of tupels containing code and embedding size for literals, clauses and global state
        :param additional_information: dictionary mapping codes of input features to their respective channel size
        """
        super().__init__()
        activation_func = F.leaky_relu
        global_pool_func = torchMax

        self.literal_code = embeddings[0][0]
        self.clause_code = embeddings[1][0]
        self.global_code = embeddings[2][0]

        self.encoder = nn.ModuleList()
        for key, embedding_size in embeddings:
            self.encoder.append(conv_map[sum(key)](embedding_size, embedding_size, 1))

        self.core = Message_Passing_Core(
            literal_embedding_size = embeddings[0][1], 
            clause_embedding_size = embeddings[1][1], 
            global_embedding_size = embeddings[2][1]
        )

        self.decoder_a = Pool_conv_sum_nonlin_pool(
            num_input_channels=Counter(dict(embeddings)) + additional_information,
            num_output_channels=1,
            output_stream_codes=[embeddings[0][0]],
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.decoder_c = Pool_conv_sum_nonlin_pool(
            num_input_channels=Counter(dict(embeddings)) + additional_information,
            num_output_channels=1,
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.apply(init_weights)
        print("Created network with embeddings =", embeddings, "and", getNumberParams(self), 'trainable paramters')

    def forward(self, population_embeddings, adjacency_matrices: List[T], timesteps: int,  additional_information: Feature_Collection = None):
        """
        Extract deep features by passing messages in the Graph network. Returns action distributions and critic values per individual.
        :param population_embeddings: List of literal, clause and global embeddings for each individual
        :param adjacency_matrices: adjacency matrix of the graph network for each individual
        :param additional_information: Feature_Collection of all additional features relevant for decoding
        """
        i = 0
        actions = []
        values = 0
        for L, C, U in population_embeddings:
            A = adjacency_matrices[i]
            A_t = A.t()
            # Encode
            L = self.encoder[0](L)
            C = self.encoder[1](C)
            U = self.encoder[2](U)

            # Process
            for i in range(timesteps):
                L, C, U = self.core(L,C,U, A, A_t)
            
            l_shape = L.shape
            L = L.view(get_full_shape(self.literal_code, L.shape))
            C = L.view(get_full_shape(self.clause_code, C.shape))
            U = L.view(get_full_shape(self.global_code, U.shape))
            
            # Decode
            action = self.decoder_a(Feature_Collection([L,C,U]).addAll(additional_information)).view(l_shape)
            value: T = self.decoder_c(Feature_Collection([L,C,U]).addAll(additional_information), pool=False)
            while(value.dim()>1):
                value=sum(-1)
            
            actions.append(action)
            values = values + value

        return actions, values

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params