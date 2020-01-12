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
    This network can process inputs of different sized dimensions.
    The outputs are a 5D or 4D tensor actor ouput and a 0D tensor as critic output. The overall structure is as follows: \n
    The inputs will be sent to a deep neural network with size num_hidden_layers and num_neurons neurons. In each layer the
    memory efficent pool_conv_sum_nonlin_pool function will be applied. The result will then be splitted into actor and critic output.
    Dimensions will be removed and the pool_conv_sum_nonlin_pool function will be applied again to form the outputs.
    """

    def __init__(self,
                num_input_channels_GxE,
                num_input_channels_PxG,
                num_input_channels_P,
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
        :param num_input_channels_GxE: Number of input channels of tensors with dimension GxE
        :param num_input_channels_PxG: Number of input channels of tensors with dimension PxG
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

        self.layers_GxE_GxE = nn.ModuleList()
        self.layers_GxE_G = nn.ModuleList()
        self.layers_GxE_E = nn.ModuleList()
        self.layers_PxG_PxG = nn.ModuleList()
        self.layers_PxG_P = nn.ModuleList()
        self.layers_PxG_G = nn.ModuleList()
        self.layers_P = nn.ModuleList()
        self.layers_1 =  nn.ModuleList()

        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):
            if layer_number == 0:
                self.layers_GxE_GxE.append(nn.Conv2d(num_input_channels_GxE, num_neurons, 1))
                self.layers_GxE_G.append(nn.Conv1d(num_input_channels_GxE, num_neurons, 1))
                self.layers_GxE_E.append(nn.Conv1d(num_input_channels_GxE, num_neurons, 1))

                self.layers_PxG_PxG.append(nn.Conv2d(num_input_channels_PxG, num_neurons, 1))
                self.layers_PxG_P.append(nn.Conv1d(num_input_channels_PxG, num_neurons, 1))
                self.layers_PxG_G.append(nn.Conv1d(num_input_channels_PxG, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_input_channels_P,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_input_channels_1,num_neurons,1))
            else:
                self.layers_GxE_GxE.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_GxE_G.append(nn.Conv1d(num_neurons, num_neurons, 1))
                self.layers_GxE_E.append(nn.Conv1d(num_neurons, num_neurons, 1))

                self.layers_PxG_PxG.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_PxG_P.append(nn.Conv1d(num_neurons, num_neurons, 1))
                self.layers_PxG_G.append(nn.Conv1d(num_neurons, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_neurons,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_neurons,num_neurons,1))

        # Generate output layers
        if self.eliminiate_genome_dimension:
            self.output_layer_actor_GxE_E = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_1 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxG_P = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_1 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)
        else:
            self.output_layer_actor_GxE_GxE = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_G = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_E = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxG_PxG = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_P = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_G = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)

        self.output_layer_critic_GxE_E = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_GxE_1= nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_PxG_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_PxG_1 = nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_1 = nn.Conv1d(num_neurons, 1, 1)

        self.apply(init_weights)

    def pool_conv_sum_nonlin_pool_5D(self, me_state: ME_State, c_GxE_GxE, c_GxE_G, c_GxE_E, 
                    c_PxG_PxG, c_PxG_P, c_PxG_G, c_P, c_1):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxE, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs
        """
        if me_state.input_PxG.dim() != 4:
            raise ValueError('Cannot handle input_PxG with dimension', me_state.input_PxG.dim())

        # Pooling
        input_GxE_G = self.global_pool_func(me_state.input_GxE, 3)[0]
        input_GxE_E = self.global_pool_func(me_state.input_GxE, 2)[0]

        input_PxG_P = self.global_pool_func(me_state.input_PxG, 3)[0]
        input_PxG_G = self.global_pool_func(me_state.input_PxG, 2)[0]

        # Conv
        me_state.input_GxE = c_GxE_GxE(me_state.input_GxE)
        input_GxE_G = c_GxE_G(input_GxE_G)
        input_GxE_E = c_GxE_E(input_GxE_E)

        me_state.input_PxG = c_PxG_PxG(me_state.input_PxG)
        input_PxG_P = c_PxG_P(input_PxG_P)
        input_PxG_G = c_PxG_G(input_PxG_G)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_1 = c_1(me_state.input_1)

        # Sum with broadcasting
        sum_PxGxE = torch.tensor(0).float()
        l = (me_state.input_PxG.unsqueeze(-1),
            me_state.input_GxE.unsqueeze(2),
            me_state.input_P.unsqueeze(-1).unsqueeze(-1),
            me_state.input_1.unsqueeze(-1).unsqueeze(-1),
            input_GxE_G.unsqueeze(2).unsqueeze(-1),
            input_GxE_E.unsqueeze(2).unsqueeze(2),
            input_PxG_P.unsqueeze(-1).unsqueeze(-1),
            input_PxG_G.unsqueeze(2).unsqueeze(-1))
        for x in l:
            sum_PxGxE = sum_PxGxE + x

        # Non-linearity
        self.activation_func(sum_PxGxE)

        # Pooling
        me_state.input_GxE = self.global_pool_func(sum_PxGxE, 2)[0]
        me_state.input_PxG = self.global_pool_func(sum_PxGxE, 4)[0]
        me_state.input_P = self.global_pool_func(me_state.input_PxG, 3)[0]
        me_state.input_1 = self.global_pool_func(me_state.input_P, 2)[0].unsqueeze(-1)

        return me_state

    def pool_conv_sum_nonlin_pool_4D(self, me_state: ME_State, c_GxE_E, c_GxE_1, c_PxG_P, c_PxG_1, c_P, c_1):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxE, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs.
        """
        if me_state.input_PxG.dim() != 3:
            raise ValueError('Cannot handle input_PxG with dimension', me_state.input_PxG.dim())

        # Genome Dimension has been removed

        # Pooling
        input_GxE_1 = self.global_pool_func(me_state.input_GxE, 2)[0]

        input_PxG_1 = self.global_pool_func(me_state.input_PxG, 2)[0]

        # Conv
        me_state.input_GxE = c_GxE_1(me_state.input_GxE)
        input_GxE_1 = c_GxE_1(input_GxE_1)

        me_state.input_PxG = c_PxG_P(me_state.input_PxG)
        input_PxG_1 = c_PxG_1(input_PxG_1)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_1 = c_1(me_state.input_1)

        # Sum with broadcasting
        sum_PxE = torch.tensor(0)
        for x in (*me_state.get_inputs(), input_GxE_1, input_PxG_1):
            sum_PxE = sum_PxE + x

        # Non-linearity
        self.activation_func(sum_PxE)

        # Pooling
        # Note that the genome dimension is ignored
        me_state.input_GxE = self.global_pool_func(sum_PxE, 2)[0]
        me_state.input_PxG = self.global_pool_func(sum_PxE, 3)[0]
        me_state.input_P = me_state.input_PxG.clone()
        me_state.input_1 = self.global_pool_func(me_state.input_P, 2)[0].unsqeeze(-1)
        
        return me_state


    def forward(self, me_state: ME_State):
        for i in range(self.num_hidden_layers + 1):
            # Pool, Conv, Sum
            me_state = self.pool_conv_sum_nonlin_pool_5D(me_state, self.layers_GxE_GxE[i], self.layers_GxE_G[i], self.layers_GxE_E[i],
                                        self.layers_PxG_PxG[i], self.layers_PxG_P[i], self.layers_PxG_G[i],
                                        self.layers_P[i], self.layers_1[i])

        action_distributions = me_state
        values = me_state.clone()

        for x in action_distributions.get_inputs():
            print("X:", x.size())

        # Eliminate dimensions before the output layers
        if self.dim_elimination_max_pooling:
            if self.eliminiate_genome_dimension:
                action_distributions.input_GxE = action_distributions.input_GxE.max(2)[0]
                action_distributions.input_PxG = action_distributions.input_PxG.max(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions.input_PxG = action_distributions.input_PxG.max(2)[0].unsqueeze(2)
            values.input_GxE = values.input_GxE.max(2)[0]
            values.input_PxG = values.input_PxG.max(3)[0]
        else:
            if self.eliminiate_genome_dimension:
                action_distributions.input_GxE = action_distributions.input_GxE.mean(2, keepdim=True)[0]
                action_distributions.input_PxG = action_distributions.input_PxG.mean(3, keepdim=True)[0]
                for x in action_distributions.get_inputs():
                    print("W:", x.size())
                if self.eliminate_population_dimension:
                    action_distributions.input_PxG = action_distributions.input_PxG.mean(2, keepdim=True)[0].unsqueeze(2)
            values.input_GxE = values.input_GxE.mean(2, keepdim=True)[0]
            values.input_PxG = values.input_PxG.mean(3, keepdim=True)[0]

        for x in action_distributions.get_inputs():
            print("Y:", x.size())

        # Calculate action output
        if self.eliminiate_genome_dimension:
            action_distributions = self.pool_conv_sum_nonlin_pool_4D(action_distributions, 
                self.output_layer_actor_GxE_E, self.output_layer_actor_GxE_1,
                self.output_layer_actor_PxG_P, self.output_layer_actor_PxG_1,
                self.output_layer_actor_P, self.output_layer_actor_1)
        else:
            action_distributions = self.pool_conv_sum_nonlin_pool_5D(action_distributions, 
            self.output_layer_actor_GxE_GxE,self.output_layer_actor_GxE_G, self.output_layer_actor_GxE_E,
            self.output_layer_actor_PxG_PxG, self.output_layer_actor_PxG_P, self.output_layer_actor_PxG_G,
            self.output_layer_actor_P, self.output_layer_actor_1)

        # Calculate value approximate
        values = self.pool_conv_sum_nonlin_pool_4D(values, self.output_layer_critic_GxE_E, self.output_layer_critic_GxE_1,
            self.output_layer_critic_PxG_P, self.output_layer_critic_PxG_1,
            self.output_layer_critic_P, self.output_layer_critic_1)

        # Sum everything up
        values = torch.cat([values.input_GxE.mean(3, keepdim=True).sum(2).view(-1),
                            values.input_PxG.mean(3, keepdim=True).sum(2).view(-1),
                            values.input_P.sum(2).view(-1),
                            values.input_1.view(-1)]).sum().unsqueeze(0)
            
        return action_distributions, values
