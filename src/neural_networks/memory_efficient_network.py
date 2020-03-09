"""
This module contains a memory efficient version of the network, allowing a list of different sized tensors as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neural_networks.memory_efficient_state import ME_State
from typing import Callable, List, Tuple
from neural_networks.utils import init_weights
T = torch.Tensor
torchMax = lambda *x: T.max(*x)[0]
conv_map = {
    0: nn.Conv1d,
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

class Memory_efficient_network(nn.Module):
    """
    This network can process inputs of different sized dimensions.
    The outputs are a 5D or 4D tensor actor ouput and a 0D tensor as critic output. The overall structure is as follows: \n
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
        print("Creating network with", num_hidden_layers, "hidden layers,", num_neurons, "neurons.")
        self.num_output_channels = num_output_channels
        self.eliminate_dimension = eliminate_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers
        self.activation_func = activation_func
        self.global_pool_func = global_pool_func

        self.layers = nn.ModuleList()
        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            self.layers.append(nn.ModuleDict())
            for input_stream in num_input_channels.keys():
                for input_code in self.get_input_stream_codes(input_stream):
                    if layer_number == 0:
                        self.layers[layer_number][str(input_code)] = conv_map[sum(input_code[3:])](num_input_channels[input_code[:3]], num_neurons, 1)
                    else:
                        self.layers[layer_number][str(input_code)] = conv_map[sum(input_code[3:])](num_neurons, num_neurons, 1)
        
        # Generate output layers
        self.output_layer_actor = nn.ModuleDict()
        for input_stream in num_input_channels.keys():
            for input_code in self.get_input_stream_codes(input_stream, self.eliminate_dimension):
                self.output_layer_actor[str(input_code)] = conv_map[sum(input_code[3:])](num_neurons, num_output_channels, 1)
        self.output_layer_critic = nn.ModuleDict()
        for input_stream in num_input_channels.keys():
            for input_code in self.get_input_stream_codes(input_stream, (0, 1, 0)):
                self.output_layer_critic[str(input_code)] = conv_map[sum(input_code[3:])](num_neurons, num_output_channels, 1)

        self.apply(init_weights)

    def pool_conv_sum_nonlin_pool(self, me_state: ME_State, conv_layers: dict, pool=True, eliminate_dimension=(0,0,0), pool_func=None):
        if not pool_func:
            pool_func = self.global_pool_func
        conv_list = dict()

        # Pooling
        input_stream_codes = list(me_state.keys())
        for input_code in input_stream_codes:
            input_stream = me_state.get(input_code)
            for sub_input_code in self.get_input_stream_codes(input_code, eliminate_dimension):
                sub_input_stream = input_stream
                for i, dim in reversed(list(enumerate(sub_input_code[3:]))):
                    if not dim:
                        if sub_input_code[i]:
                            sub_input_stream = pool_func(sub_input_stream, 2+i, sub_input_stream.dim()<=3)
                        elif sub_input_stream.dim()>3:
                            sub_input_stream = sub_input_stream.squeeze(2+i)
                conv_list[sub_input_code] = sub_input_stream
        me_state = None
        #  Conv
        for input_code, input_stream in conv_list.items():
            conv_list[input_code] = conv_layers[str(input_code)](input_stream)

        # Sum with broadcasting
        sum_PxGxE = torch.tensor(0).float()
        for input_code in list(conv_list.keys()):
            input_stream = conv_list.pop(input_code)
            for i, dim in enumerate(input_code[3:]):
                if not dim and input_stream.dim()<5:
                    input_stream = input_stream.unsqueeze(2+i)
            sum_PxGxE = sum_PxGxE + input_stream
        
        if len(conv_list) > 0:
            raise ValueError('Not all input streams were used! ', conv_list)

        # Non-linearity
        sum_PxGxE = self.activation_func(sum_PxGxE)

        if(not pool):
            return sum_PxGxE.view(*sum_PxGxE.shape[0:2], *tuple(filter(lambda x: x>1, sum_PxGxE.shape[2:])))

        # Pooling
        me_state = ME_State()
        for input_code in input_stream_codes:
            input_stream = sum_PxGxE
            for i, dim in enumerate(input_code):
                if not dim:
                    input_stream = self.global_pool_func(input_stream, 2+i, True)
            me_state.store(input_stream)
        
        return me_state

    def forward(self, me_state: ME_State):
        for i in range(self.num_hidden_layers + 1):
            # Pool, Conv, Sum
            me_state = self.pool_conv_sum_nonlin_pool(me_state, self.layers[i])
            torch.cuda.empty_cache()

        pool_func = torchMax if self.dim_elimination_max_pooling else T.mean

        # Calculate action output
        action_distributions = self.pool_conv_sum_nonlin_pool(me_state, self.output_layer_actor, pool=False, eliminate_dimension=self.eliminate_dimension, pool_func=pool_func)

        # Calculate value approximate
        values = self.pool_conv_sum_nonlin_pool(me_state, self.output_layer_critic, eliminate_dimension=(0,1,0), pool_func=pool_func)

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

        return action_distributions, values

    def get_input_stream_codes(self, input_code: Tuple[int], eliminate_dimension = (0,0,0)) -> List[Tuple[int]]:
        """
        Calculates all input stream the given input devides into and returns their encodings
        :param input_code: the encoding of the input stream
        :param eliminate_dimension: tuple that encodes all dimensions that are eliminated
        """
        inputs = []
        code = tuple(1*np.greater(input_code, eliminate_dimension))
        inputs = [input_code+code]
        if sum(code) > 1:
            for i, dim in enumerate(code):
                if dim and not eliminate_dimension[i]:
                    inputs.append(input_code + tuple(0 if j==i else code[j] for j in range(3)))
        return inputs
