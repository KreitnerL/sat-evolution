from __future__ import annotations
import torch
from torch import Tensor as T
from torch.autograd import Variable
from typing import List, Tuple

class ME_State:
    """
    Wrapper class for a dictionary of different sized Tensors.
    The key of each tensor is a tupel encoding the tensors size, e.g. (1,0,1) belongs to a Tensor with size Px1xE.
    Each Tensor has the form BxCxPxGxE, the variables being:\n
    B: Batchsize \n
    C: Channelsize \n
    P: Populationsize \n
    G: Genomesize \n
    E: Equationsize
    """
    def __init__(self, inputs: List[T] = [], memory: List[T] = []):
        """
        Creates a new dictionary, that maps every input code to its belonging input.
        :param inputs: A list of tensors of size BxCxPxGxE
        """
        self.memory = dict()
        self.storeMemory(memory)
        self.input_streams = dict()
        for input_stream in inputs:
            self.store(input_stream)

    def items(self):
        """
        returns key, value pairs for all input streams
        """
        return self.input_streams.items()

    def keys(self):
        """
        returns all input codes of the dictionary
        """
        return self.input_streams.keys()

    def values(self):
        """
        returns all input streams of the dictionary
        """
        l = list(self.input_streams.values())
        l.sort(key=lambda x: x.size())
        return l

    def get(self, input_code: Tuple[int]) -> T:
        """
        Given an input code, returns the belonging input stream.
        :param input_code: A tupel encoding the tensor size, e.g. (1,0,1) belongs to a Tensor with size Px1xE
        """
        return self.input_streams[input_code]

    def getMemory(self) -> List[T]:
        return self.memory.values()

    def store(self, input_stream: T, overwrite=False):
        """
        Adds the given Tensor to the dictionary.
        :param input_stream: the tensor that should be stored
        :param overwrite: (Optional) If True, the previous entry will be overwritten. The entries will be concatenated along the channels dimension otherwise. Default=False 
        """
        code = self.getCode(input_stream.size())
        if overwrite or code not in self.input_streams:
            self.input_streams[code] = input_stream
        else:
            self.input_streams[code] = torch.cat([self.input_streams[code], input_stream], 1)

    def storeMemory(self, memory: List[T]):
        """
        Adds the given Tensor to the memory.
        :param input_stream: the tensor that should be stored
        :param overwrite: (Optional) If True, the previous entry will be overwritten. The entries will be concatenated along the channels dimension otherwise. Default=False 
        """
        for input_stream in memory:
            code = self.getCode(input_stream.size())
            if code not in self.memory:
                self.memory[code] = input_stream
            else:
                self.memory[code] = torch.cat([self.memory[code], input_stream], 1)

    def apply_fn(self, fn) -> ME_State:
        """
        Returns a copy of the current state with the given function applied to all Tensors in the dictionary.
        Be aware, that performing actions that change the size of a Tensor will lead to an inconsistent state!
        """
        return ME_State(
            [fn(array) if array is not None else None for array in self.values()],
            [fn(value) for value in self.memory.values()]
        )

    def clone(self) -> ME_State:
        """
        Returns a copy of the given state by cloning all stored Tensors
        """
        return self.apply_fn(lambda array: array.clone().detach())

    def to_cuda_variable(self) -> ME_State:
        """
        Applies the cuda() function to all stored tensors
        """
        return self.apply_fn(lambda array: Variable(array).cuda())

    def detach(self) -> ME_State:
        """
        Detaches all stored Tensors from the GPU
        """
        return self.apply_fn(lambda array: array.detach())

    def __str__(self):
        return 'ME_State: ' + str([array.size() for array in self.values()] + str([array.size() for array in self.memory]))

    def getCode(self, size: Tuple[int]) -> Tuple[int]:
        """
        Return the store code for a given Tensor size.
        :param size: Tuple of form [batch, channel, P, G, E]
        """
        return tuple(1 if dim>1 else 0 for dim in size[2:])

def concat(me_states: List[ME_State]) -> ME_State:
    """
    Concatenates the given states by concatenating all Tensors that have the same size and returns the resulting state.
    """
    inputs_array = list(zip(*tuple([me_state.values() for me_state in me_states])))
    memory_array = list(zip(*tuple([me_state.getMemory() for me_state in me_states])))
    return ME_State([torch.cat(inputs) for inputs in inputs_array], [torch.cat(inputs) for inputs in memory_array])