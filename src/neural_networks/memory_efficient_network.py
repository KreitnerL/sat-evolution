"""
This module contains a memory efficient version of the network, allowing a list of different sized tensors as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.memory_efficient_state import ME_State
from neural_networks.pool_conv_sum_nonlin_pool import Pool_conv_sum_nonlin_pool
from neural_networks.utils import init_weights
from collections import Counter
from typing import Tuple, List
from solvers.encoding import ProblemInstanceEncoding
NUM_DIMENSIONS = ProblemInstanceEncoding.NUM_DIMENSIONS
T = torch.Tensor
torchMax = lambda *x: torch.max(*x)[0]

class Memory_efficient_network(nn.Module):
    """
    This network can process features of different sized dimensions without broadcasting, leading to memory efficency.
    The outputs are a 2D tensor actor ouput and a 0D tensor as critic output. The overall structure is as follows: \n
    The inputs will be sent to a deep neural network with size num_hidden_layers and num_neurons neurons. In each layer the
    memory efficent pool_conv_sum_nonlin_pool function will be applied. The result will then be splitted into actor and critic output.
    Dimensions will be removed and the pool_conv_sum_nonlin_pool function will be applied again to form the outputs.
    """
    def __init__(self,
                num_input_channels: tuple,
                num_output_channels: int,
                dim_elimination_max_pooling=False,
                eliminate_dimension=[0]+[1]*(NUM_DIMENSIONS-1),
                num_hidden_layers=1,
                num_neurons=32,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(torchMax) = torchMax):
        """
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels: Number of output channels for the actor component
        :param static_features: List of the stream_codes that will be used for long term inferences
        :param dim_elimination_max_pooling: If true, dimensions will be removed via max pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        self.num_output_channels = num_output_channels
        # For output eliminate all dimensions except Population dim
        self.eliminate_dimension = eliminate_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers


        all_feature_dim, self.memory_dim, self.dynamic_features, self.static_features = num_input_channels
        a = set(self.static_features) & set(self.memory_dim.keys())

        #####################################
        # Generate input / hidden layers    #
        #####################################
        self.dynamic_layer = nn.ModuleList()
        self.static_layer = nn.ModuleList()
        for layer_number in range(num_hidden_layers+1):
            self.dynamic_layer.append(Pool_conv_sum_nonlin_pool(
                num_input_channels={x: all_feature_dim[x] if layer_number==0 else num_neurons for x in self.dynamic_features},
                num_output_channels=num_neurons,
                output_stream_codes=self.dynamic_features,
                activation_func=activation_func,
                global_pool_func=global_pool_func))
            self.static_layer.append(Pool_conv_sum_nonlin_pool(
                num_input_channels={x: all_feature_dim[x] if layer_number==0 else num_neurons for x in self.static_features},
                num_output_channels=num_neurons if layer_number < num_hidden_layers else next(iter(self.memory_dim.values())),
                output_stream_codes=self.static_features if layer_number < num_hidden_layers else set(self.static_features) & set(self.memory_dim.keys()),
                activation_func=activation_func,
                global_pool_func=global_pool_func))

        #####################################
        # Generate output layers            #
        #####################################
        # Create Actor and Critic output layer
        self.output_layer_actor_critic = Pool_conv_sum_nonlin_pool(
            num_input_channels={x: num_neurons for x in self.dynamic_features},
            num_output_channels=num_output_channels*2,
            eliminate_dimension=self.eliminate_dimension,
            activation_func=activation_func,
            global_pool_func=global_pool_func)
        # Create memory output layer
        self.memory_output = Pool_conv_sum_nonlin_pool(
            num_input_channels={x: num_neurons for x in self.dynamic_features},
            num_output_channels=next(iter(self.memory_dim.values())),
            output_stream_codes=[code for code in self.memory_dim.keys() if code not in self.static_features],
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.apply(init_weights)
        print("Created network with", num_hidden_layers, "hidden layers,", num_neurons, "neurons and", getNumberParams(self), 'trainable paramters')

    def forward(self, input_t: ME_State):
        torch.cuda.empty_cache()
        memory_t = input_t.getMemory()

        # Concat input and memory
        input_t.addAll(memory_t)

        # Devide in tempory and static features:
        dynamic_state = input_t
        static_state = ME_State()
        dynamic_state = ME_State()
        for code in self.static_features:
            static_state.store(input_t.get(code))
        for code in self.dynamic_features:
            dynamic_state.store(input_t.get(code))

        # Apply Input and hidden layers
        for i in range(self.num_hidden_layers + 1):
            dynamic_state = self.dynamic_layer[i](dynamic_state)
            static_state = self.static_layer[i](static_state)

        # Calculate action and critic output. Action and critic should be calculated independently (Form = Bx2xP)
        pool_func = torchMax if self.dim_elimination_max_pooling else T.mean
        action_critic = self.output_layer_actor_critic(dynamic_state, pool=False, pool_func=pool_func)
        action_distributions, values = action_critic.narrow(1,0, self.num_output_channels), action_critic.narrow(1,self.num_output_channels-1, self.num_output_channels)
        for dim in range(len(self.eliminate_dimension)-sum(self.eliminate_dimension)+1):
            values = values.sum(-1)

        # Calculate memory(t+1)
        if self.memory_output:
            memory_t = self.memory_output(dynamic_state)
            memory_t.addAll(static_state)
            memory_t = memory_t.apply_fn(torch.tanh)

        # detach memory for truncated backpropagation through time
        return action_distributions, values, memory_t.cpu().detach()

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params
