import torch
import torch.nn as nn
import numpy as np
from neural_networks.memory_efficient_state import ME_State
from typing import List, Tuple
import torch.nn.functional as F
import itertools

T = torch.Tensor
torchMax = lambda *x: T.max(*x)[0]
conv_map = {
    0: nn.Conv1d,
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

class Pool_conv_sum_nonlin_pool(nn.Module):
    """
    Submodule that takes different sized feature tensors and applies pooling, convolution, summing with broadcasting, a non-linearity and pooling.
    All features are convoluted by their very own conv-layer, so that they do not have to be concatenated, leading to a more memory efficent implementation.
    """

    def __init__(
        self, 
        num_input_channels: dict, 
        num_output_channels: int, 
        output_stream_codes: List[int] = None,
        eliminate_dimension: Tuple[int] = (0,0,0),
        activation_func: type(F.leaky_relu) = F.leaky_relu,
        global_pool_func: type(torchMax) = torchMax):
        """
        Generates a submodule that can take a ME_State (list of features) with the given feature dimensions and returns either a concatenated tensor or a ME_State.
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels:  Dictionary that assigns a number of output channels to each input code
        :param eliminate_dimension: boolean tupel that encodes for each dimension whether it should be removed
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        if eliminate_dimension != (0,0,0):
            num_input_channels_new = dict()
            for code, channels in num_input_channels.items():
                code = tuple(1*np.greater(code, eliminate_dimension))
                num_input_channels_new[code] = num_input_channels_new.get(code, 0) + channels
            num_input_channels = num_input_channels_new
        # Calculate all sub stream code per input
        self.input_stream_codes = {code: get_input_stream_codes(code) for code in num_input_channels.keys()}
        self.output_stream_codes = output_stream_codes if output_stream_codes is not None else self.input_stream_codes.keys()
        self.activation_func = activation_func
        self.global_pool_func = global_pool_func
        self.eliminate_dimension = eliminate_dimension

        # Generate Conv Layers
        self.layers = nn.ModuleDict()
        for l in self.input_stream_codes.values():
            for input_code in l:
                self.layers[str(input_code)] = conv_map[sum(input_code[3:])](num_input_channels[input_code[:3]], num_output_channels, 1)

    def forward(self, me_state: ME_State, pool=True, pool_func=None):
        if not pool_func:
            pool_func = self.global_pool_func

        if self.eliminate_dimension != (0,0,0):
            new = ME_State()
            for code, x in me_state.items():
                for i, dim in enumerate(np.logical_and(code, self.eliminate_dimension)):
                    if dim:
                        x = self.global_pool_func(x, 2+i, True)
                new.store(x)
            me_state = new
        self.checkInput(me_state)
        
        conv_list = dict()

        # Pooling
        for input_code, sub_input_code_list in self.input_stream_codes.items():
            input_stream = me_state.get(input_code)
            for sub_input_code in sub_input_code_list:
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
            conv_list[input_code] = self.layers[str(input_code)](input_stream)

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
        for input_code in self.output_stream_codes:
            input_stream = sum_PxGxE
            for i, dim in enumerate(input_code):
                if not dim:
                    input_stream = self.global_pool_func(input_stream, 2+i, True)
            me_state.store(input_stream)
        
        return me_state

    def checkInput(self, me_state: ME_State):
        a = list(set(me_state.keys()))
        b = list(set(self.input_stream_codes.keys()))
        if a != b:
            a.sort()
            b.sort()
            raise ValueError(str(a) + ' != ' + str(b))


def get_input_stream_codes(input_code: Tuple[int]) -> List[Tuple[int]]:
        """
        Calculates all input stream the given input devides into and returns their encodings
        :param input_code: the encoding of the input stream
        :param eliminate_dimension: tuple that encodes all dimensions that are eliminated
        """
        if sum(input_code) <= 1:
            return [input_code+input_code]

        def _constraints(code: Tuple[int]) -> bool:
            for i, x in enumerate(code):
                if x and not input_code[i]:
                    return False
            return sum(code)>0

        return list(map(lambda x: input_code + x, filter(_constraints, itertools.product((0,1), repeat=len(input_code)))))
