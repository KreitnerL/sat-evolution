from torch import Tensor as T

class ME_State:
    """
    Encodes a collection of different sized Tensors
    Each Tensor has the form BatchSize x Channels x Dimension(indivdual for every input)
    """
    def __init__(self, input_GxCx2: T, input_PxGx2: T, input_P: T, input_G: T, input_1: T):
        self.input_GxCx2 = input_GxCx2
        self.input_PxGx2 = input_PxGx2
        self.input_P = input_P
        self.input_G = input_G
        self.input_1 = input_1

    def get_inputs(self):
        return self.input_GxCx2, self.input_PxGx2, self.input_P, self.input_G, self.input_1

    def clone(self):
        return ME_State( self.input_GxCx2.clone(), self.input_PxGx2.clone(), self.input_P.clone(), self.input_G.clone(), self.input_1.clone())