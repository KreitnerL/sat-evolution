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
                num_input_channels_PxE,
                num_input_channels_P,
                num_input_channels_1,
                num_output_channels,
                eliminate_genome_dimension=True,
                eliminate_clause_dimension=True,
                eliminate_population_dimension=False,
                dim_elimination_max_pooling=False,
                num_hidden_layers=1,
                num_neurons=128,
                activation_func: type(F.leaky_relu) = F.leaky_relu,
                global_pool_func: type(T.max) = T.max):
        """
        :param num_input_channels_GxE: Number of input channels of tensors with dimension GxE
        :param num_input_channels_PxG: Number of input channels of tensors with dimension PxG
        :param num_input_channels_PxE: Number of input channels of tensors with dimension PxE
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
        self.eliminate_genome_dimension = eliminate_genome_dimension
        self.eliminate_clause_dimension = eliminate_clause_dimension
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
        self.layers_PxE_PxE = nn.ModuleList()
        self.layers_PxE_P = nn.ModuleList()
        self.layers_PxE_E = nn.ModuleList()
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

                self.layers_PxE_PxE.append(nn.Conv2d(num_input_channels_PxE, num_neurons, 1))
                self.layers_PxE_P.append(nn.Conv1d(num_input_channels_PxE, num_neurons, 1))
                self.layers_PxE_E.append(nn.Conv1d(num_input_channels_PxE, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_input_channels_P,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_input_channels_1,num_neurons,1))
            else:
                self.layers_GxE_GxE.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_GxE_G.append(nn.Conv1d(num_neurons, num_neurons, 1))
                self.layers_GxE_E.append(nn.Conv1d(num_neurons, num_neurons, 1))

                self.layers_PxG_PxG.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_PxG_P.append(nn.Conv1d(num_neurons, num_neurons, 1))
                self.layers_PxG_G.append(nn.Conv1d(num_neurons, num_neurons, 1))

                self.layers_PxE_PxE.append(nn.Conv2d(num_neurons, num_neurons, 1))
                self.layers_PxE_P.append(nn.Conv1d(num_neurons, num_neurons, 1))
                self.layers_PxE_E.append(nn.Conv1d(num_neurons, num_neurons, 1))

                self.layers_P.append(nn.Conv1d(num_neurons,num_neurons,1))
                self.layers_1.append(nn.Conv1d(num_neurons,num_neurons,1))

        # Generate output layers
        if self.eliminate_genome_dimension:
            self.output_layer_actor_GxE_E = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_1 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxG_P = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_1 = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxE_PxE = nn.Conv2d(num_neurons,1,1)
            self.output_layer_actor_PxE_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_PxE_E = nn.Conv1d(num_neurons,1,1)
            
            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)
        else:
            self.output_layer_actor_GxE_GxE = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_G = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_GxE_E = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxG_PxG = nn.Conv2d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_P = nn.Conv1d(num_neurons, 1, 1)
            self.output_layer_actor_PxG_G = nn.Conv1d(num_neurons, 1, 1)

            self.output_layer_actor_PxE_PxE = nn.Conv2d(num_neurons,1,1)
            self.output_layer_actor_PxE_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_PxE_E = nn.Conv1d(num_neurons,1,1)

            self.output_layer_actor_P = nn.Conv1d(num_neurons,1,1)
            self.output_layer_actor_1 = nn.Conv1d(num_neurons,1,1)

        self.output_layer_critic_GxE_E = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_GxE_1= nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_PxG_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_PxG_1 = nn.Conv1d(num_neurons, 1, 1)

        self.output_layer_critic_PxE_PxE = nn.Conv2d(num_neurons,1,1)
        self.output_layer_critic_PxE_P = nn.Conv1d(num_neurons,1,1)
        self.output_layer_critic_PxE_E = nn.Conv1d(num_neurons,1,1)

        self.output_layer_critic_P = nn.Conv1d(num_neurons, 1, 1)
        self.output_layer_critic_1 = nn.Conv1d(num_neurons, 1, 1)

        self.apply(init_weights)

    def pool_conv_sum_nonlin_pool_4D(self, me_state: ME_State, c_GxE_GxE, c_GxE_G, c_GxE_E, 
                    c_PxG_PxG, c_PxG_P, c_PxG_G, c_PxE_PxE, c_PxE_P, c_PxE_E, c_P, c_1, pool=True):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxE, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs
        """
        if me_state.input_PxG.dim() != 4:
            raise ValueError('Cannot handle input_PxG with dimension', me_state.input_PxG.size())

        # Pooling
        input_GxE_G = self.global_pool_func(me_state.input_GxE, 3)[0]
        input_GxE_E = self.global_pool_func(me_state.input_GxE, 2)[0]

        input_PxG_P = self.global_pool_func(me_state.input_PxG, 3)[0]
        input_PxG_G = self.global_pool_func(me_state.input_PxG, 2)[0]

        input_PxE_P = self.global_pool_func(me_state.input_PxE, 3)[0]
        input_PxE_E = self.global_pool_func(me_state.input_PxE, 2)[0]

        # Conv
        me_state.input_GxE = c_GxE_GxE(me_state.input_GxE)
        input_GxE_G = c_GxE_G(input_GxE_G)
        input_GxE_E = c_GxE_E(input_GxE_E)

        me_state.input_PxG = c_PxG_PxG(me_state.input_PxG)
        input_PxG_P = c_PxG_P(input_PxG_P)
        input_PxG_G = c_PxG_G(input_PxG_G)

        me_state.input_PxE = c_PxE_PxE(me_state.input_PxE)
        input_PxE_P = c_PxE_P(input_PxE_P)
        input_PxE_E = c_PxE_E(input_PxE_E)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_1 = c_1(me_state.input_1)

        # Sum with broadcasting
        sum_PxGxE = torch.tensor(0).float()
        l = [me_state.input_PxG.unsqueeze(-1),
            me_state.input_GxE.unsqueeze(2),
            me_state.input_PxE.unsqueeze(-2),
            me_state.input_P.unsqueeze(-1).unsqueeze(-1),
            me_state.input_1.unsqueeze(-1).unsqueeze(-1),
            input_GxE_G.unsqueeze(2).unsqueeze(-1),
            input_GxE_E.unsqueeze(2).unsqueeze(2),
            input_PxG_P.unsqueeze(-1).unsqueeze(-1),
            input_PxG_G.unsqueeze(2).unsqueeze(-1),
            input_PxE_P.unsqueeze(-1).unsqueeze(-1),
            input_PxE_E.unsqueeze(2).unsqueeze(2)]
        input_GxE_G = input_GxE_E = input_PxG_P = input_PxG_G = input_PxE_P = input_PxE_E = me_state = None
        while len(l) != 0:
            sum_PxGxE = sum_PxGxE + l.pop()

        # Non-linearity
        sum_PxGxE = self.activation_func(sum_PxGxE)

        # Pooling
        me_state = ME_State()
        me_state.input_GxE = self.global_pool_func(sum_PxGxE, 2)[0]
        me_state.input_PxG = self.global_pool_func(sum_PxGxE, 4)[0]
        me_state.input_PxE = self.global_pool_func(sum_PxGxE, 3)[0]
        me_state.input_P = self.global_pool_func(me_state.input_PxG, 3)[0]
        me_state.input_1 = self.global_pool_func(me_state.input_P, 2)[0].unsqueeze(-1)

        return me_state

    def pool_conv_sum_nonlin_pool_3D(self, me_state: ME_State, c_GxE_E, c_GxE_1, c_PxG_P, c_PxG_1, c_PxE_PxE, c_PxE_P, c_PxE_E, c_P, c_1, pool=True):
        """
        Takes a ME_State of Tensors of size Batch x Channels x Dimension( Dimension being e.g. GxE, P, ...) applies the global_pool_function,
        convolutes every single array (hence memory efficient) sums everything up (broadcasting), applies the global activation function and pools again for the outputs.
        """
        if me_state.input_PxG.dim() != 3:
            raise ValueError('Cannot handle input_PxG with dimension', me_state.input_PxG.size())
        # Genome Dimension has been removed

        # Pooling
        input_GxE_1 = self.global_pool_func(me_state.input_GxE, 2, keepdim=True)[0]

        input_PxG_1 = self.global_pool_func(me_state.input_PxG, 2, keepdim=True)[0]

        input_PxE_P = self.global_pool_func(me_state.input_PxE, 3)[0]
        input_PxE_E = self.global_pool_func(me_state.input_PxE, 2)[0]

        # Conv
        me_state.input_GxE = c_GxE_E(me_state.input_GxE)
        input_GxE_1 = c_GxE_1(input_GxE_1)

        me_state.input_PxG = c_PxG_P(me_state.input_PxG)
        input_PxG_1 = c_PxG_1(input_PxG_1)

        me_state.input_PxE = c_PxE_PxE(me_state.input_PxE)
        input_PxE_P = c_PxE_P(input_PxE_P)
        input_PxE_E = c_PxE_E(input_PxE_E)

        me_state.input_P = c_P(me_state.input_P)
        me_state.input_1 = c_1(me_state.input_1)

        # Sum with broadcasting
        sum_PxE = torch.tensor(0)
        l = [me_state.input_GxE.unsqueeze(2),
            me_state.input_PxG.unsqueeze(-1),
            me_state.input_P.unsqueeze(-1),
            me_state.input_PxE,
            me_state.input_1.unsqueeze(-1),
            input_GxE_1.unsqueeze(-1),
            input_PxG_1.unsqueeze(-1),
            input_PxE_P.unsqueeze(-1),
            input_PxE_E.unsqueeze(2)]
        input_GxE_1 = input_PxG_1 = input_PxE_P = input_PxE_E = me_state = None
        while len(l) != 0:
            sum_PxE = sum_PxE + l.pop()

        # Non-linearity
        sum_PxE = self.activation_func(sum_PxE)

        if(not pool):
            return sum_PxE
        # Pooling (Note that the genome dimension does not exist)
        me_state = ME_State()
        me_state.input_GxE = self.global_pool_func(sum_PxE, 2)[0]
        me_state.input_PxG = self.global_pool_func(sum_PxE, 3)[0]
        me_state.input_PxE = sum_PxE
        me_state.input_P = me_state.input_PxG.clone()
        me_state.input_1 = self.global_pool_func(me_state.input_P, 2, keepdim=True)[0]
        
        return me_state


    def forward(self, me_state: ME_State):
        for i in range(self.num_hidden_layers + 1):
            # Pool, Conv, Sum
            me_state = self.pool_conv_sum_nonlin_pool_4D(
                me_state=me_state,
                c_GxE_GxE=self.layers_GxE_GxE[i],
                c_GxE_G=self.layers_GxE_G[i],
                c_GxE_E=self.layers_GxE_E[i],
                c_PxG_PxG=self.layers_PxG_PxG[i],
                c_PxG_P=self.layers_PxG_P[i],
                c_PxG_G=self.layers_PxG_G[i],
                c_PxE_PxE=self.layers_PxE_PxE[i],
                c_PxE_P=self.layers_PxE_P[i],
                c_PxE_E=self.layers_PxE_E[i],
                c_P=self.layers_P[i],
                c_1=self.layers_1[i])
            torch.cuda.empty_cache()

        action_distributions = me_state
        values = me_state.clone()

        # Eliminate dimensions before the output layers
        if self.dim_elimination_max_pooling:
            if self.eliminate_genome_dimension:
                action_distributions.input_GxE = action_distributions.input_GxE.max(2)[0]
                action_distributions.input_PxG = action_distributions.input_PxG.max(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions.input_PxG = action_distributions.input_PxG.max(2)[0].unsqueeze(2)
                    action_distributions.input_PxE = action_distributions.input_PxE.max(2)[0].unsqueeze(2)
            values.input_GxE = values.input_GxE.max(2)[0]
            values.input_PxG = values.input_PxG.max(3)[0]
        else:
            if self.eliminate_genome_dimension:
                action_distributions.input_GxE = action_distributions.input_GxE.mean(2)
                action_distributions.input_PxG = action_distributions.input_PxG.mean(3)
                if self.eliminate_population_dimension:
                    action_distributions.input_PxG = action_distributions.input_PxG.mean(2).unsqueeze(2)
                    action_distributions.input_PxE = action_distributions.input_PxE.mean(2).unsqueeze(2)
            values.input_GxE = values.input_GxE.mean(2)
            values.input_PxG = values.input_PxG.mean(3)

        # Calculate action output
        if self.eliminate_genome_dimension:
            action_distributions = self.pool_conv_sum_nonlin_pool_3D(
                me_state=action_distributions,
                c_GxE_E=self.output_layer_actor_GxE_E,
                c_GxE_1=self.output_layer_actor_GxE_1,
                c_PxG_P=self.output_layer_actor_PxG_P,
                c_PxG_1=self.output_layer_actor_PxG_1,
                c_PxE_PxE=self.output_layer_actor_PxE_PxE,
                c_PxE_P=self.output_layer_actor_PxE_P,
                c_PxE_E=self.output_layer_actor_PxE_E,
                c_P=self.output_layer_actor_P,
                c_1=self.output_layer_actor_1,
                pool=False)
        else:
            action_distributions = self.pool_conv_sum_nonlin_pool_4D(
                me_state=action_distributions, 
                c_GxE_GxE=self.output_layer_actor_GxE_GxE,
                c_GxE_G=self.output_layer_actor_GxE_G,
                c_GxE_E=self.output_layer_actor_GxE_E,
                c_PxG_PxG=self.output_layer_actor_PxG_PxG,
                c_PxG_P=self.output_layer_actor_PxG_P,
                c_PxG_G=self.output_layer_actor_PxG_G,
                c_PxE_PxE=self.output_layer_actor_PxE_PxE[i],
                c_PxE_P=self.output_layer_actor_PxE_P[i],
                c_PxE_E=self.output_layer_actor_PxE_E[i],
                c_P=self.output_layer_actor_P,
                c_1=self.output_layer_actor_1,
                pool=False)
        if self.eliminate_clause_dimension:
            action_distributions = action_distributions.mean(-1)

        # Calculate value approximate
        values = self.pool_conv_sum_nonlin_pool_3D(
            me_state=values,
            c_GxE_E=self.output_layer_critic_GxE_E,
            c_GxE_1=self.output_layer_critic_GxE_1,
            c_PxG_P=self.output_layer_critic_PxG_P,
            c_PxG_1=self.output_layer_critic_PxG_1,
            c_PxE_PxE=self.output_layer_critic_PxE_PxE,
            c_PxE_P=self.output_layer_critic_PxE_P,
            c_PxE_E=self.output_layer_critic_PxE_E,
            c_P=self.output_layer_critic_P,
            c_1=self.output_layer_critic_1)

        # Sum everything up
        values = sum((values.input_GxE.sum(2),
                        values.input_PxG.sum(2),
                        values.input_PxE.sum(2).sum(2),
                        values.input_P.sum(2),
                        values.input_1.sum(2)), 2).view(-1)
        return action_distributions, values
