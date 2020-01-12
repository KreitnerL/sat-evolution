from torch import Tensor as T
from torch.autograd import Variable

class ME_State:
    """
    Encodes a collection of different sized Tensors
    Each Tensor has the form BatchSize x Channels x Dimension(indivdual for every input)
    """
    def __init__(self, input_GxE: T, input_PxG: T, input_P: T, input_G: T, input_1: T):
        self.input_GxE = input_GxE
        self.input_PxG = input_PxG
        self.input_P = input_P
        self.input_G = input_G
        self.input_1 = input_1

    def get_inputs(self):
        return self.input_GxE, self.input_PxG, self.input_P, self.input_G, self.input_1

    def clone(self):
        return ME_State( self.input_GxE.clone(), self.input_PxG.clone(), self.input_P.clone(), self.input_G.clone(), self.input_1.clone())

    def apply_fn(self, fn):
        return ME_State(*tuple(fn(array) if array is not None else None for array in self.get_inputs()))

    def to_cuda_variable(self):
        return self.apply_fn(lambda array: Variable(array).cuda())

    def detach(self):
        return self.apply_fn(lambda array: array.detach())