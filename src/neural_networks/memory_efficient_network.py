"""
This module contains a memory efficient version of the network, allowing a list of different sized tensors as input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neural_networks.memory_efficient_state import ME_State
from typing import Callable, List
from neural_networks.utils import init_weights

T = torch.Tensor

class Memory_efficient_network(nn.Module):
    """
    This network can process inputs of different sized dimensions (1xGxCx2, PxGx1x2, Px1x1x1, 1xGx1x1, 1x1x1x1).
    The outputs are a 5D or 4D tensor actor ouput and a 0D tensor as critic output. The overall structure is as follows: \n
    The inputs will be sent to a deep neural network with size num_hidden_layers and num_neurons neurons. In each layer the
    memory efficent pool_conv_sum_nonlin_pool function will be applied. The result will then be splitted into actor and critic output.
    Dimensions will be removed and the pool_conv_sum_nonlin_pool function will be applied again to form the outputs.
    """

    def __init__(self,
                num_input_channels_GxCx2,
                num_input_channels_PxGx2,
                num_input_channels_P,
                num_input_channels_G,
                num_input_channels_1,
                num_output_channels,
                eliminate_genome_dimension=True,
                eliminate_population_dimension=False,
                dim_elimination_max_pooling=False,
                num_hidden_layers=1,
                num_neurons=128,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(T.max) = T.max):
        """
        :param num_input_channels_GxCx2: Number of input channels of tensors with dimension GxCx2
        :param num_input_channels_PxGx2: Number of input channels of tensors with dimension PxGx2
        :param num_input_channels_P: Number of input channels of tensors with dimension P
        :param num_input_channels_G: Number of input channels of tensors with dimension G
        :param num_input_channels_1: Number of input channels of tensors with dimension 1
        :param num_output_channels: Number of output channels for the actor component
        :param eliminate_genome_dimension: Whether to eliminate the 4th dimension of the actor output
        :param eliminate_population_dimension: Whether to also eliminate the 3rd dimension of the actor output
        :param dim_eliminiation_max_pooling: Whether to use max- or mean-pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param activation_func: Activation function used as non-linearity.
        :param global_pool_func: Pooling function used to reduce the sum to the output dimensions.
        """
        super().__init__()
        self.num_output_channels = num_output_channels
        self.eliminiate_genome_dimension = eliminate_genome_dimension
        self.eliminate_population_dimension = eliminate_population_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling
        self.num_hidden_layers = num_hidden_layers

        self.activation_func = activation_func
        self.global_pool_func = global_pool_func

        # self.input_norm = nn.BatchNorm2d(num_input_channels)

        self.layers_GxCx2_GxCx2 = nn.ModuleList()
        self.layers_GxCx2_Gx2 = nn.ModuleList()
        self.layers_GxCx2_Cx2 = nn.ModuleList()
        self.layers_PxGx2_PxGx2 = nn.ModuleList()
        self.layers_PxGx2_Px2 = nn.ModuleList()
        self.layers_PxGx2_Gx2 = nn.ModuleList()
        self.layers_P = nn.ModuleList()
        self.layers_G = nn.ModuleList()
        self.layers_1 =  nn.ModuleList()

        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            if layer_number == 0:
                self.layers_GxCx2_GxCx2.append(nn.Conv3d(num_input_channels_GxCx2, num_neurons, 1))
                self.layers_GxCx2_Gx2.append(nn.Conv2d(num_input_channels_GxCx2, num_neurons, 1))
                self.layers_GxCx2_Cx2.append(nn.Conv2d(num_input_channels_GxCx2, num_neurons, 1))

                self.layers_PxGx2_PxGx2.append(nn.Conv3d(num_input_channels_PxGx2, num_neurons, 1))
                self.layers_PxGx2_Px2.append(nn.Conv2d(num_input_channels_PxGx2, num_neurons, 1))
                self.layers_PxGx2_Gx2.append(nn.Conv2d(num_input_channels_PxGx2, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_input_channels_P,num_neurons,1))
                self.layers_G.append(nn.Conv1d(num_input_channels_G,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_input_channels_1,num_neurons,1))
            else:
                self.layers_GxCx2_GxCx2.append(nn.Conv3d(num_neurons, num_neurons, 1))
                self.layers_GxCx2_Gx2.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_GxCx2_Cx2.append(nn.Conv2d(num_neurons, num_neurons, 1))

                self.layers_PxGx2_PxGx2.append(nn.Conv3d(num_neurons, num_neurons, 1))
                self.layers_PxGx2_Px2.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_PxGx2_Gx2.append(nn.Conv2d(num_neurons, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_input_channels_P,num_neurons,1))
                self.layers_G.append(nn.Conv1d(num_input_channels_G,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_input_channels_1,num_neurons,1))

        # Generate output layers
        if self.eliminiate_genome_dimension:
            self.output_layer_actor_GxCx2_Cx2 = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_GxCx2_2 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxGx2_Px2 = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_PxGx2_2 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_G = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)
        else:
            self.output_layer_actor_GxCx2_GxCx2 = nn.Conv3d(num_neurons, 1, 1)
            self.output_layer_actor_GxCx2_Gx2 = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_GxCx2_Cx2 = nn.Conv2d(num_neurons, 1, 1)

            self.output_layer_actor_PxGx2_PxGx2 = nn.Conv3d(num_neurons, 1, 1)
            self.output_layer_actor_PxGx2_Px2 = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_PxGx2_Gx2 = nn.Conv2d(num_neurons, 1, 1)

            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_G = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)

        self.output_layer_critic_GxCx2_Cx2 = nn.Conv2d(num_neurons, 1, 1)
        self.output_layer_critic_GxCx2_2= nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_PxGx2_Px2 = nn.Conv2d(num_neurons, 1, 1)
        self.output_layer_critic_PxGx2_2 = nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_G = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_1 = nn.Conv1d(num_neurons, 1, 1)

        self.apply(init_weights)

    def pool_conv_sum_nonlin_pool_5D(self, me_state: ME_State, c_GxCx2_GxCx2, c_GxCx2_Gx2, c_GxCx2_Cx2, 
                    c_PxGx2_PxGx2, c_PxGx2_Px2, c_PxGx2_Gx2, c_P, c_G, c_1):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxCx2, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs
        """
        if me_state.input_PxGx2.dim() != 5:
            raise ValueError('Cannot handle input_PxGx2 with dimension', me_state.input_PxGx2.dim())

        # Pooling
        input_GxCx2_Gx2 = self.global_pool_func(me_state.input_GxCx2, 3)[0]
        input_GxCx2_Cx2 = self.global_pool_func(me_state.input_GxCx2, 2)[0]

        input_PxGx2_Px2 = self.global_pool_func(me_state.input_PxGx2, 3)[0]
        input_PxGx2_Gx2 = self.global_pool_func(me_state.input_PxGx2, 2)[0]

        # Conv
        me_state.input_GxCx2 = c_GxCx2_GxCx2(me_state.input_GxCx2)
        input_GxCx2_Gx2 = c_GxCx2_Gx2(input_GxCx2_Gx2)
        input_GxCx2_Cx2 = c_GxCx2_Cx2(input_GxCx2_Cx2)

        me_state.input_PxGx2 = c_PxGx2_PxGx2(me_state.input_PxGx2)
        input_PxGx2_Px2 = c_PxGx2_Px2(input_PxGx2_Px2)
        input_PxGx2_Gx2 = c_PxGx2_Gx2(input_PxGx2_Gx2)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_G = c_G(me_state.input_G)
        me_state.input_1 = c_1(me_state.input_G)

        # Sum with broadcasting
        sum_PxGxCx2 = torch.tensor(0)
        for x in me_state.get_inputs():
            sum_PxGxCx2 += x

        # Non-linearity
        self.activation_func(sum_PxGxCx2)

        # Pooling
        me_state.input_GxCx2 = self.global_pool_func(sum_PxGxCx2, 2)[0]
        me_state.input_PxGx2 = self.global_pool_func(sum_PxGxCx2, 4)[0]
        tmp_PxG = self.global_pool_func(me_state.input_PxGx2, 4)[0]
        me_state.input_P = self.global_pool_func(tmp_PxG, 3)[0]
        me_state.input_G = self.global_pool_func(tmp_PxG, 2)[0]
        me_state.input_1 = self.global_pool_func(me_state.input_G, 2)[0]

        return me_state

    def pool_conv_sum_nonlin_pool_4D(self, me_state: ME_State, c_GxCx2_Cx2, c_GxCx2_2, c_PxGx2_Px2, c_PxGx2_2, c_P, c_G, c_1):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxCx2, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs.
        """
        if me_state.input_PxGx2.dim() != 4:
            raise ValueError('Cannot handle input_PxGx2 with dimension', me_state.input_PxGx2.dim())

        # Genome Dimension has been removed

        # Pooling
        input_GxCx2_2 = self.global_pool_func(me_state.input_GxCx2, 2)[0]

        input_PxGx2_2 = self.global_pool_func(me_state.input_PxGx2, 2)[0]

        # Conv
        me_state.input_GxCx2 = c_GxCx2_2(me_state.input_GxCx2)
        input_GxCx2_2 = c_GxCx2_2(input_GxCx2_2)

        me_state.input_PxGx2 = c_PxGx2_Px2(me_state.input_PxGx2)
        input_PxGx2_2 = c_PxGx2_2(input_PxGx2_2)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_G = c_G(me_state.input_G)
        me_state.input_1 = c_1(me_state.input_G)

        # Sum with broadcasting
        sum_PxCx2 = torch.tensor(0)
        for x in me_state.get_inputs():
            sum_PxCx2 += x

        # Non-linearity
        self.activation_func(sum_PxCx2)

        # Pooling
        # Note that the genome dimension is ignored
        me_state.input_GxCx2 = self.global_pool_func(sum_PxCx2, 2)[0]
        me_state.input_PxGx2 = sum_PxCx2
        tmp_Px2 = self.global_pool_func(sum_PxCx2, 3)[0]
        me_state.input_P = self.global_pool_func(tmp_Px2, 3)[0]
        me_state.input_G = self.global_pool_func(me_state.input_P, 2)[0]
        me_state.input_1 = me_state.input_G.clone()
        
        return me_state


    def forward(self, me_state: ME_State):
        for i in range(self.num_hidden_layers + 1):
            # Pool, Conv, Sum
            me_state = self.pool_conv_sum_nonlin_pool_5D(me_state, self.layers_GxCx2_GxCx2[i], self.layers_GxCx2_Gx2[i], self.layers_GxCx2_Cx2[i],
                                        self.layers_PxGx2_PxGx2[i], self.layers_PxGx2_Px2[i], self.layers_PxGx2_Gx2[i],
                                        self.layers_P[i], self.layers_G[i], self.layers_1[i])

        action_distributions = me_state
        values = me_state.clone()

        # Eliminate dimensions before the output layers
        if self.dim_elimination_max_pooling:
            if self.eliminiate_genome_dimension:
                action_distributions.input_GxCx2 = action_distributions.input_GxCx2.max(2)[0]
                action_distributions.input_PxGx2 = action_distributions.input_PxGx2.max(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions.input_PxGx2 = action_distributions.input_PxGx2.max(2)[0].unsqueeze(2)
            values.input_GxCx2 = values.input_GxCx2.max(2)[0]
            values.input_PxGx2 = values.input_GxCx2.max(3)[0]
        else:
            if self.eliminiate_genome_dimension:
                action_distributions.input_GxCx2 = action_distributions.input_GxCx2.mean(2)[0]
                action_distributions.input_PxGx2 = action_distributions.input_PxGx2.mean(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions.input_PxGx2 = action_distributions.input_PxGx2.mean(2)[0].unsqueeze(2)
            values.input_GxCx2 = values.input_GxCx2.mean(2)[0]
            values.input_PxGx2 = values.input_GxCx2.mean(3)[0]

        # Calculate action output
        if self.eliminiate_genome_dimension:
            action_distributions = self.pool_conv_sum_nonlin_pool_4D(action_distributions, 
                self.output_layer_actor_GxCx2_Cx2, self.output_layer_actor_GxCx2_2,
                self.output_layer_actor_PxGx2_Px2, self.output_layer_actor_PxGx2_2,
                self.output_layer_actor_P, self.output_layer_actor_G, self.output_layer_actor_1)
        else:
            action_distributions = self.pool_conv_sum_nonlin_pool_5D(action_distributions, 
            self.output_layer_actor_GxCx2_GxCx2,self.output_layer_actor_GxCx2_Gx2, self.output_layer_actor_GxCx2_Cx2,
            self.output_layer_actor_PxGx2_PxGx2, self.output_layer_actor_PxGx2_Px2, self.output_layer_actor_PxGx2_Gx2,
            self.output_layer_actor_P, self.output_layer_actor_G, self.output_layer_actor_1)

        # Calculate value approximate
        values = self.pool_conv_sum_nonlin_pool_4D(values, self.output_layer_critic_GxCx2_Cx2, self.output_layer_critic_GxCx2_2,
            self.output_layer_critic_PxGx2_Px2, self.output_layer_critic_PxGx2_2,
            self.output_layer_critic_P, self.output_layer_critic_G, self.output_layer_critic_1)

        # Sum everything up
        values = torch.cat([values.input_GxCx2.mean(3).sum(2).view(-1),
                            values.input_PxGx2.mean(3).sum(2).view(-1),
                            values.input_P.sum(2).view(-1),
                            values.input_G.sum(2).view(-1),
                            values.input_1.view(-1)]).sum().unsqueeze(0)
            
        return action_distributions, values
