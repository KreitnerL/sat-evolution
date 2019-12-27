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
    Sub layer that broadcasts and sums its inputs of different dimensions (1xGxCx2, PxGx1x2, Px1x1x1, 1xGx1x1, 1x1x1x1),
    applies a non-linearity and pools it to get the outputs.
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
        :param num_input_channels: Number of input channel
        :param num_output_channels: Number of output channels for the actor component
        :param eliminate_length_dimension: Whether to eliminate the 4th dimension of the actor output
        :param eliminate_population_dimension: Whether to also eliminate the 3rd dimension of the actor output
        :param dim_eliminiation_max_pooling: Whether to use max- or mean-pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param eliminate_length_dimension: Whether to eliminate
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

        self.layers_GxCx2 = nn.ModuleList()
        self.layers_PxGx2 = nn.ModuleList()
        self.layers_P = nn.ModuleList()
        self.layers_G = nn.ModuleList()
        self.layers_1 =  nn.ModuleList()

        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            if layer_number == 0: 
                self.layers_GxCx2.append(nn.Conv3d(num_input_channels_GxCx2*3,num_neurons,1))
                self.layers_PxGx2.append(nn.Conv3d(num_input_channels_PxGx2*3,num_neurons,1))
                self.layers_P.append(nn.Conv1d(num_input_channels_P,num_neurons,1))
                self.layers_G.append(nn.Conv1d(num_input_channels_G,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_input_channels_1,num_neurons,1))
            else:
                self.layers_GxCx2.append(nn.Conv3d(num_neurons*3,num_neurons,1))
                self.layers_PxGx2.append(nn.Conv3d(num_neurons*3,num_neurons,1))
                self.layers_P.append(nn.Conv1d(num_neurons,num_neurons,1))
                self.layers_G.append(nn.Conv1d(num_neurons,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_neurons,num_neurons,1))

        # Generate output layers
        if self.eliminiate_genome_dimension:
            self.output_layer_actor_GxCx2 = nn.Conv2d(num_neurons * 2, 1, 1)
            self.output_layer_actor_PxGx2 = nn.Conv2d(num_neurons * 2, 1, 1)
            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_G = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)
        else:
            self.output_layer_actor_GxCx2 = nn.Conv3d(num_neurons * 3, 1, 1)
            self.output_layer_actor_PxGx2 = nn.Conv3d(num_neurons * 3, 1, 1)
            self.output_layer_actor_P = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_G = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_GxCx2 = nn.Conv2d(num_neurons * 2, 1, 1)
        self.output_layer_critic_PxGx2 = nn.Conv2d(num_neurons * 2, 1, 1)
        self.output_layer_critic_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_G = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_1 = nn.Conv1d(num_neurons, 1, 1)

        self.apply(init_weights)

    def add_broadcasted(self, me_state: ME_State):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxCx2, P, ...) applies the global_pool_function 
        and broadcasts global information along the Channels dimension
        """
        if me_state.input_PxGx2.dim() == 5:
            # Pooling
            input_Gx1x2 = self.global_pool_func(me_state.input_GxCx2, 3)[0].unsqueeze(3)
            input_1xCx2 = self.global_pool_func(me_state.input_GxCx2, 2)[0].unsqueeze(2)

            input_Px1x2 = self.global_pool_func(me_state.input_PxGx2, 3)[0].unsqueeze(3)
            input_1xGx2 = self.global_pool_func(me_state.input_PxGx2, 2)[0].unsqueeze(2)

            # Concat
            input_Gx1x2.expand_as(me_state.input_GxCx2)
            input_1xCx2.expand_as(me_state.input_GxCx2)

            input_Px1x2.expand_as(me_state.input_PxGx2)
            input_1xGx2.expand_as(me_state.input_PxGx2)

            # Concatenate original x and broadcasted information along feature dimension
            me_state.input_GxCx2 = torch.cat([me_state.input_GxCx2, input_Gx1x2, input_1xCx2], 1)
            me_state.input_PxGx2 = torch.cat([me_state.input_PxGx2, input_Px1x2, input_1xGx2], 1)
        elif me_state.input_PxGx2.dim() == 4:
            # Genome Dimension has been removed

            # Pooling
            input_1x1x2_from_GxCx2 = self.global_pool_func(me_state.input_GxCx2, 2)[0].unsqueeze(2)

            input_1x1x2_from_PxGx2 = self.global_pool_func(me_state.input_PxGx2, 2)[0].unsqueeze(2)

            # Concat
            input_1x1x2_from_GxCx2.expand_as(me_state.input_GxCx2)

            input_1x1x2_from_PxGx2.expand_as(me_state.input_PxGx2)

            # Concatenate original x and broadcasted information along feature dimension
            me_state.input_GxCx2 = torch.cat([me_state.input_GxCx2, input_1x1x2_from_GxCx2], 1)
            me_state.input_PxGx2 = torch.cat([me_state.input_PxGx2, input_1x1x2_from_PxGx2], 1)
        else:
            raise ValueError('Cannot handle input_PxGx2 with dimension', me_state.input_PxGx2.dim())

        return me_state


    def forward(self, me_state: ME_State):
        for i in range(self.num_hidden_layers + 1):
            # Pool, Replicate, Concat
            me_state = self.add_broadcasted(me_state)
            # Conv, Activate
            me_state.input_GxCx2 = self.activation_func(self.layers_GxCx2[i](me_state.input_GxCx2))
            me_state.input_PxGx2 = self.activation_func(self.layers_PxGx2[i](me_state.input_PxGx2))
            me_state.input_P = self.activation_func(self.layers_P[i](me_state.input_P))
            me_state.input_G = self.activation_func(self.layers_G[i](me_state.input_G))
            me_state.input_1 = self.activation_func(self.layers_1[i](me_state.input_G))

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
            if self.eliminiate_length_dimension:
                action_distributions.input_GxCx2 = action_distributions.input_GxCx2.mean(2)[0]
                action_distributions.input_PxGx2 = action_distributions.input_PxGx2.mean(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions.input_PxGx2 = action_distributions.input_PxGx2.mean(2)[0].unsqueeze(2)
            values.input_GxCx2 = values.input_GxCx2.mean(2)[0]
            values.input_PxGx2 = values.input_GxCx2.mean(3)[0]

        # Calculate action output
        action_distributions = self.add_broadcasted(action_distributions)
        action_distributions.input_GxCx2 = self.output_layer_actor_GxCx2(action_distributions.input_GxCx2)
        action_distributions.input_PxGx2 = self.output_layer_actor_PxGx2(action_distributions.input_PxGx2)
        action_distributions.input_P = self.output_layer_actor_P(action_distributions.input_P)
        action_distributions.input_G = self.output_layer_actor_G(action_distributions.input_G)
        action_distributions.input_1 = self.output_layer_actor_1(action_distributions.input_1)

        # Calculate value approximate
        values = self.add_broadcasted(values)
        values.input_GxCx2 = self.output_layer_critic_GxCx2(values.input_GxCx2)
        values.input_PxGx2 = self.output_layer_critic_PxGx2(values.input_PxGx2)
        values.input_P = self.output_layer_critic_P(values.input_P)
        values.input_G = self.output_layer_critic_G(values.input_G)
        values.input_1 = self.output_layer_critic_1(values.input_1)

        # Sum everything up
        values = torch.cat([values.input_GxCx2.mean(3).sum(2).view(-1),
                            values.input_PxGx2.mean(3).sum(2).view(-1),
                            values.input_P.sum(2).view(-1),
                            values.input_G.sum(2).view(-1),
                            values.input_1.view(-1)]).sum().unsqueeze(0)
            
        return action_distributions, values
