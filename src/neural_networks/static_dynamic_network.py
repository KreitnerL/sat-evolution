"""
This module contains a memory efficient version of the network, allowing a list of different sized tensors as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from neural_networks.feature_collection import Feature_Collection
from neural_networks.pool_conv_sum_nonlin_pool import Pool_conv_sum_nonlin_pool
from neural_networks.utils import init_weights, getNumberParams
from collections import Counter
from typing import Tuple, List
from solvers.encoding import ProblemInstanceEncoding
NUM_DIMENSIONS = ProblemInstanceEncoding.NUM_DIMENSIONS
T = torch.Tensor
torchMax = lambda *x: torch.max(*x)[0]

class Static_Dynamic_Network(nn.Module):
    """
    This network consists out of two steps. First, it uses theoretical reasoning over "static" input (input that does not change for this problem + population independent) to extract deep features.
    It then reports the infered features to the practical reasoning part, that makes a plan for each individual how to improve their solution. The network outputs a critic value and action distributions
    (combined actor-critic model), as well as a memory output that will be used as again as input in the next timestep. This allows the network to work towards a solution over many generations.
    """
    def __init__(self,
                num_input_channels: tuple,
                num_output_channels: int,
                dim_elimination_max_pooling=False,
                eliminate_dimension=[0]+[1]*(NUM_DIMENSIONS-1),
                num_hidden_layers: Tuple[int]=(1,0),
                num_neurons=32,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(torchMax) = torchMax):
        """
        :param num_input_channels: Dictionary that assigns a number of channels to each input code
        :param num_output_channels: Number of output channels for the actor component
        :param dim_elimination_max_pooling: If true, dimensions will be removed via max pooling
        :param num_hidden_layers: Number of hidden layers layers for the theoretical reason part and the practical reasoning part
        :param num_neurons: Number of neurons / filters per conv layer
        :param activation_func: Activation function used as non-linearity
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions
        """
        super().__init__()
        self.dummy = T([])
        self.dummy.requires_grad = True
        self.num_output_channels = num_output_channels
        # For output eliminate all dimensions except Population dim
        self.eliminate_dimension = eliminate_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers

        all_feature_dim, self.memory_dim, self.practical_features, self.theoretical_features, self.report = num_input_channels

        #####################################
        # Generate input / hidden layers    #
        #####################################
        self.practical_layer = nn.ModuleList()
        self.theoretical_layer = nn.ModuleList()
        for layer_number in range(num_hidden_layers[1]+1):
            self.theoretical_layer.append(Pool_conv_sum_nonlin_pool(
                num_input_channels={x: all_feature_dim[x] if layer_number==0 else num_neurons for x in self.theoretical_features},
                num_output_channels=num_neurons if layer_number < num_hidden_layers[1] else max(self.memory_dim.values()),
                output_stream_codes=self.theoretical_features if layer_number < num_hidden_layers[1] else set(self.theoretical_features) & set(self.memory_dim.keys()),
                activation_func=activation_func,
                global_pool_func=global_pool_func))

        for code in self.report:
            all_feature_dim[code] = self.memory_dim[code]

        for layer_number in range(num_hidden_layers[0]+1):
            self.practical_layer.append(Pool_conv_sum_nonlin_pool(
                num_input_channels={x: all_feature_dim[x] if layer_number==0 else num_neurons for x in self.practical_features},
                num_output_channels=num_neurons,
                output_stream_codes=self.practical_features,
                activation_func=activation_func,
                global_pool_func=global_pool_func))


        #####################################
        # Generate output layers            #
        #####################################
        # Create Actor and Critic output layer
        self.output_layer_actor_critic = Pool_conv_sum_nonlin_pool(
            num_input_channels={x: num_neurons for x in self.practical_features},
            num_output_channels=num_output_channels*2,
            eliminate_dimension=self.eliminate_dimension,
            activation_func=activation_func,
            global_pool_func=global_pool_func)
        # Create memory output layer
        self.memory_output = Pool_conv_sum_nonlin_pool(
            num_input_channels={x: num_neurons for x in self.practical_features},
            num_output_channels=next(iter(self.memory_dim.values())),
            output_stream_codes=[code for code in self.memory_dim.keys() if code not in self.theoretical_features],
            activation_func=activation_func,
            global_pool_func=global_pool_func)

        self.apply(init_weights)
        print("Created network with", num_hidden_layers, "hidden layers,", num_neurons, "neurons and", getNumberParams(self), 'trainable paramters')

    def forward(self, input_t: Feature_Collection):
        """
        :param input_t: Feature_Collection with all relevant features as well as the current memory
        Returns a critic value and an action distributions (combined actor-critic model), as well as the new memory.
        """
        torch.cuda.empty_cache()
        # Concat input and memory
        input_t.addAll(input_t.getMemory())
        # Use checkpointing for network to prevent storing itermediate activations (see https://pytorch.org/docs/stable/checkpoint.html)
        # Note:
        # 1) Checkpoint only allows a tuple of tensors as inputs / outputs. 
        # 2) If all inputs have required_grad=False it will not calculate the grad_fn. => Insert Dummy tensor with required_grad=True
        action_distributions, values, *memory_t = cp.checkpoint(self.bottleneck, self.dummy, *input_t.values())
        return action_distributions, values, Feature_Collection(memory_t)

    def bottleneck(self, _dummy, *features):
        input_t = Feature_Collection(features)
        # Devide in tempory and theoretical features:
        practical_state = input_t
        theoretical_state = Feature_Collection()
        practical_state = Feature_Collection()
        for code in self.theoretical_features:
            theoretical_state.store(input_t.get(code))
        for code in self.practical_features:
            practical_state.store(input_t.get(code))

        # Apply Input and hidden layers
        for i in range(self.num_hidden_layers[1] + 1):
            theoretical_state = self.theoretical_layer[i](theoretical_state)

        # Use results from theroretical reasoning
        for code in self.report:
            practical_state.store(theoretical_state.get(code), overwrite=True)

        for i in range(self.num_hidden_layers[0] + 1):
            practical_state = self.practical_layer[i](practical_state)

        # Calculate action and critic output. Action and critic should be calculated independently (Form = Bx2xP)
        pool_func = torchMax if self.dim_elimination_max_pooling else T.mean
        action_critic = self.output_layer_actor_critic(practical_state, pool=False, pool_func=pool_func)
        action_distributions, values = action_critic.narrow(1,0, self.num_output_channels), action_critic.narrow(1,self.num_output_channels-1, self.num_output_channels)
        while(values.dim()>1):
            values = values.sum(-1)

        # Calculate memory(t+1)
        if self.memory_output:
            memory_t = self.memory_output(practical_state).addAll(theoretical_state).apply_fn(torch.tanh).cpu()

        # detach memory for truncated backpropagation through time
        return (action_distributions, values) + tuple(memory_t.values())
