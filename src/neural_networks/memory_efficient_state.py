import torch
from torch import Tensor as T
from torch.autograd import Variable
from typing import List, Tuple

class ME_State:
    """
    Encodes a collection of different sized Tensors
    Each Tensor has the form BatchSize x Channels x Dimension(indivdual for every input)
    """
    def __init__(self, input_GxE: T, input_PxG: T, input_P: T, input_1: T):
        self.input_GxE = input_GxE
        self.input_PxG = input_PxG
        self.input_P = input_P
        self.input_1 = input_1

    def get_inputs(self):
        return self.input_GxE, self.input_PxG, self.input_P, self.input_1

    def apply_fn(self, fn):
        return ME_State(*tuple(fn(array) if array is not None else None for array in self.get_inputs()))

    def clone(self):
        return self.apply_fn(lambda array: array.clone().detach())

    def to_cuda_variable(self):
        return self.apply_fn(lambda array: Variable(array).cuda())

    def detach(self):
        return self.apply_fn(lambda array: array.detach())

    def __str__(self):
        return 'ME_State: ' + str([array.size() for array in self.get_inputs()])

def concat(me_states: List[ME_State]) -> ME_State:
    inputs_array = list(zip(*tuple([me_state.get_inputs() for me_state in me_states])))
    combined_states: List[T] = []
    for inputs in inputs_array:
        combined_states.append(torch.cat(inputs))
    # pylint: disable=no-value-for-parameter
    return ME_State(*tuple(combined_states))