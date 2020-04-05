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
from typing import Tuple
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
                num_input_channels: Tuple[dict],
                num_output_channels: int,
                dim_elimination_max_pooling=False,
                num_hidden_layers=1,
                num_neurons=128,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(torchMax) = torchMax):
        """
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels: Number of output channels for the actor component
        :param dim_elimination_max_pooling: If true, dimensions will be removed via max pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        self.num_output_channels = num_output_channels
        # For output eliminate all dimensions except Population dim
        self.eliminate_dimension = tuple([0]+[1]*(NUM_DIMENSIONS-1))
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers

        self.memory_dim = num_input_channels[1]
        neurons_dict = Counter(dict.fromkeys(num_input_channels[0], num_neurons))
        
        self.layers = nn.ModuleList()
        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            if layer_number == 0:
                self.layers.append(Pool_conv_sum_nonlin_pool(
                    num_input_channels=num_input_channels[0],
                    num_output_channels=num_neurons,
                    activation_func=activation_func,
                    global_pool_func=global_pool_func))
            else:
                self.layers.append(Pool_conv_sum_nonlin_pool(
                    num_input_channels=neurons_dict,
                    num_output_channels=num_neurons,
                    activation_func=activation_func,
                    global_pool_func=global_pool_func))
        
        # Generate output layers
        self.output_layer_actor_critic = Pool_conv_sum_nonlin_pool(
            num_input_channels=neurons_dict+neurons_dict,
            num_output_channels=num_output_channels*2,
            eliminate_dimension=self.eliminate_dimension,
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        # Only create layer if memory is propagated
        self.memory_output = Pool_conv_sum_nonlin_pool(
            num_input_channels=neurons_dict,
            num_output_channels=list(self.memory_dim.values())[0],
            output_stream_codes=self.memory_dim.keys(),
            activation_func=activation_func,
            global_pool_func=global_pool_func) if self.memory_dim else None

        self.apply(init_weights)
        print("Created network with", num_hidden_layers, "hidden layers,", num_neurons, "neurons and", getNumberParams(self), 'trainable paramters')

    def forward(self, input_t: ME_State):
        torch.cuda.empty_cache()
        memory_t = input_t.getMemory()

        # Concat input and memory
        input_t.addAll(memory_t)

        for i in range(self.num_hidden_layers + 1):
            input_t = self.layers[i](input_t)

        pool_func = torchMax if self.dim_elimination_max_pooling else T.mean

        double_input = input_t.clone()
        double_input.addAll(input_t)
        # Calculate action and critic output (Form = Bx2xP)
        action_critic = self.output_layer_actor_critic(double_input, pool=False, pool_func=pool_func)

        action_distributions, values = action_critic.select(1,0).unsqueeze(1), action_critic.select(1,1)

        # Sum everything up
        values = values.sum(-1)

        # Calculate memory(t+1)
        if self.memory_output:
            memory_t = self.memory_output(input_t).apply_fn(torch.tanh)

        # detach memory for truncated backpropagation through time
        return action_distributions, values, memory_t.detach()

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params
