"""
This module contains a memory efficient version of the network, allowing a list of different sized tensors as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.memory_efficient_state import ME_State
from neural_networks.pool_conv_sum_nonlin_pool import Pool_conv_sum_nonlin_pool
from neural_networks.utils import init_weights
torchMax = lambda *x: T.max(*x)[0]
T = torch.Tensor

class Memory_efficient_network(nn.Module):
    """
    This network can process features of different sized dimensions without broadcasting, leading to memory efficency.
    The outputs are a 2D tensor actor ouput and a 0D tensor as critic output. The overall structure is as follows: \n
    The inputs will be sent to a deep neural network with size num_hidden_layers and num_neurons neurons. In each layer the
    memory efficent pool_conv_sum_nonlin_pool function will be applied. The result will then be splitted into actor and critic output.
    Dimensions will be removed and the pool_conv_sum_nonlin_pool function will be applied again to form the outputs.
    """
    def __init__(self,
                num_input_channels: dict,
                num_output_channels: int,
                eliminate_dimension=(0, 1, 1),
                dim_elimination_max_pooling=False,
                num_hidden_layers=1,
                num_neurons=128,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(torchMax) = torchMax):
        """
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels: Number of output channels for the actor component
        :param eliminate_dimension: boolean tupel that encodes for each dimension whether it should be removed
        :param dim_elimination_max_pooling: If true, dimensions will be removed via max pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        self.num_output_channels = num_output_channels
        self.eliminate_dimension = eliminate_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers
        self.memory_dim = dict()

        neurons_dict: dict = dict.fromkeys(num_input_channels, num_neurons)

        self.layers = nn.ModuleList()
        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            if layer_number == 0:
                self.layers.append(Pool_conv_sum_nonlin_pool(
                    num_input_channels=num_input_channels,
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
        self.output_layer_actor = Pool_conv_sum_nonlin_pool(
            num_input_channels=neurons_dict,
            num_output_channels=num_output_channels,
            eliminate_dimension=eliminate_dimension,
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.output_layer_critic = Pool_conv_sum_nonlin_pool(
            num_input_channels=neurons_dict,
            num_output_channels=num_output_channels,
            eliminate_dimension=(0,1,0),
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.apply(init_weights)
        print("Created network with", num_hidden_layers, "hidden layers,", num_neurons, "neurons and", getNumberParams(self), 'trainable paramters')

    def forward(self, input_t: ME_State):
        memory_t = input_t.getMemory()
        if not memory_t:
            memory_t = [torch.zeros(1,1, p, g, e).to_cuda_variable() for p,g,e in self.memory_dim]

        # Concat input and hiddent state
        for value in memory_t:
            input_t.store(value)

        for i in range(self.num_hidden_layers + 1):
            input_t = self.layers[i](input_t)
            torch.cuda.empty_cache()

        pool_func = torchMax if self.dim_elimination_max_pooling else T.mean

        # Calculate action output
        action_distributions = self.output_layer_actor(input_t, pool=False, pool_func=pool_func)

        # Calculate value approximate
        values = self.output_layer_critic(input_t, pool_func=pool_func)

        # Sum everything up
        l = []
        for input_code, input_stream in values.items():
            for i, dim in enumerate(input_code):
                if dim:
                    input_stream = input_stream.sum(2)
                else:
                    input_stream = input_stream.squeeze(2)
            l.append(input_stream)
        values = sum(l, 2).view(-1)

        return action_distributions, values, memory_t

def getNumberParams(network):
    num_params = 0
    for p in network.parameters():
        num_params += p.data.view(-1).size(0)
    return num_params
