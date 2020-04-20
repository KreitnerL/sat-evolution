import torch
import torch.nn as nn
import numpy as np
from neural_networks.feature_collection import Feature_Collection
from typing import List, Tuple
import torch.nn.functional as F
import itertools
from solvers.encoding import ProblemInstanceEncoding
NUM_DIMENSIONS = ProblemInstanceEncoding.NUM_DIMENSIONS

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
    Submodule that takes different sized feature tensors and applies prior pooling, convolution, summing with broadcasting, a non-linearity and pooling, where both poolings are optional.
    All features are convoluted by their very own conv-layer, so that they do not have to be concatenated, yielding a more memory efficent implementation.
    """

    def __init__(
        self, 
        num_input_channels: dict, 
        num_output_channels: int, 
        output_stream_codes: List[int] = None,
        eliminate_dimension: Tuple[int] = tuple([0]*NUM_DIMENSIONS),
        prior_pooling: bool = False,
        activation_func: type(F.leaky_relu) = F.leaky_relu,
        global_pool_func: type(torchMax) = torchMax):
        """
        Generates a submodule that can take a Feature_Collection (list of features) with the given feature dimensions and returns either a concatenated tensor or a Feature_Collection.
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels: number of output channels
        :param output_stream_codes: List of codes that fixate the dimension of the output tensors. If not set, the network will return the same input dimensions as the input
        :param eliminate_dimension: Boolean tupel that encodes for each dimension whether it should be removed
        :param prior_pooling: True if one wants to extract global features for each input tensor
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        # Elimination of dimension lead to another input. Calculate the new input dimensions + channels. Assumes that all inputs were pooled from the same tensor before.
        if sum(eliminate_dimension) > 0:
            num_input_channels = dict.fromkeys({tuple(1*np.greater(code, eliminate_dimension)) for code in num_input_channels}, list(num_input_channels.values())[0])
        # Calculate all sub stream codes per input
        self.prior_pooling = prior_pooling
        self.input_stream_codes = {code: self.get_input_stream_codes(code) for code in num_input_channels.keys()}
        self.output_stream_codes = output_stream_codes if output_stream_codes is not None else self.input_stream_codes.keys()
        self.activation_func = activation_func
        self.global_pool_func = global_pool_func
        self.eliminate_dimension = eliminate_dimension

        # Generate Conv Layers
        self.layers = nn.ModuleDict()
        for l in self.input_stream_codes.values():
            for input_code in l:
                self.layers[str(input_code)] = conv_map[sum(input_code[NUM_DIMENSIONS:])](num_input_channels[input_code[:NUM_DIMENSIONS]], num_output_channels, 1)

    def forward(self, feature_collection: Feature_Collection, pool=True, pool_func=None):
        if not pool_func:
            pool_func = self.global_pool_func

        # Remove unwanted dimensions
        if sum(self.eliminate_dimension) > 0:
            new = Feature_Collection()
            for code, x in feature_collection.items():
                for i, dim in enumerate(np.logical_and(code, self.eliminate_dimension)):
                    if dim:
                        x = self.global_pool_func(x, 2+i, True)
                new.store(x, overwrite=True)
            feature_collection = new
        self.checkInput(feature_collection)
        
        conv_list = dict()

        # Pooling
        for input_code, sub_input_code_list in self.input_stream_codes.items():
            input_stream = feature_collection.get(input_code)
            for sub_input_code in sub_input_code_list:
                sub_input_stream = input_stream
                for i, dim in reversed(list(enumerate(sub_input_code[NUM_DIMENSIONS:]))):
                    if not dim and sub_input_code[i]:
                        sub_input_stream = pool_func(sub_input_stream, 2+i, sub_input_stream.dim()<=3)
                sub_input_stream = sub_input_stream.view(sub_input_stream.shape[0], sub_input_stream.shape[1], *(x for x in sub_input_stream.shape[2:] if x>1))
                if sub_input_stream.dim()<=2:
                    sub_input_stream = sub_input_stream.unsqueeze(-1)
                conv_list[sub_input_code] = sub_input_stream
        feature_collection = None

        # Conv
        for input_code, input_stream in conv_list.items():
            conv_list[input_code] = self.layers[str(input_code)](input_stream)

        # Sum with broadcasting
        sum_PxGxE = torch.tensor(0).float()
        for input_code in list(conv_list.keys()):
            input_stream = conv_list.pop(input_code)
            input_stream = input_stream.view(*get_full_shape(input_code[NUM_DIMENSIONS:], input_stream.shape))
            sum_PxGxE = sum_PxGxE + input_stream
        
        if len(conv_list) > 0:
            raise ValueError('Not all input streams were used! ', conv_list)

        # Non-linearity
        sum_PxGxE = self.activation_func(sum_PxGxE)

        if(not pool):
            return sum_PxGxE.view(*sum_PxGxE.shape[0:2], *tuple(filter(lambda x: x>1, sum_PxGxE.shape[2:])))

        # Pooling
        feature_collection = Feature_Collection()
        for input_code in self.output_stream_codes:
            input_stream = sum_PxGxE
            for i, dim in enumerate(input_code):
                if not dim:
                    input_stream = self.global_pool_func(input_stream, 2+i, True)
            feature_collection.store(input_stream)
        
        return feature_collection

    def checkInput(self, feature_collection: Feature_Collection):
        """
        Thows an exception if the given feature_collection does not have same feature dimensions (after eliminate dimensions!) as stated in the initialization
        """
        a = set(feature_collection.keys())
        b = set(self.input_stream_codes.keys())
        if a != b:
            a, b = list(a), list(b)
            a.sort()
            b.sort()
            raise ValueError(str(a) + ' != ' + str(b))


    def get_input_stream_codes(self, input_code: Tuple[int]) -> List[Tuple[int]]:
        """
        Calculates all input streams, the given input devides into and returns their encodings.
        :param input_code: the encoding of the input stream
        """
        if sum(input_code) <= 1 or not self.prior_pooling:
            return [input_code+input_code]

        def _constraints(code: Tuple[int]) -> bool:
            for i, x in enumerate(code):
                if x and not input_code[i]:
                    return False
            return sum(code)>0

        return list(map(lambda x: input_code + x, filter(_constraints, itertools.product((0,1), repeat=len(input_code)))))

def get_full_shape(code: Tuple[int], shape: Tuple[int]) -> List[int]:
    """
    Given the code of and the squeezed shape of a tensor, returns the full shape of said tensor before it was squeezed.
    """
    s = list(shape[:2])
    i = 2
    for dim in code:
        if dim:
            s.append(shape[i])
            i+=1
        else:
            s.append(1)
    return s
