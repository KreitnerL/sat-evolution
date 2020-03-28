"""
This module contains strategies for performing mutation.
"""

from abc import abstractmethod
from random import uniform

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from neural_networks.memory_efficient_state import ME_State

from neural_networks.memory_efficient_network import Memory_efficient_network
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy


class IndividualMutationControl(PPOStrategy):

    """
    Outputs mutation rates for each individual in the population
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='baseline',
                 learning_rate=1e-5,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=128,
                 batch_size=32,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 fixed_std_deviation=-1,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5):

        num_output_channels = 2

        network = Memory_efficient_network(
            encoding_strategy.num_channels(),
            num_output_channels,
            eliminate_dimension=(0,1,0),
            dim_elimination_max_pooling=dim_elimination_max_pooling,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).cuda()

        super().__init__(network,
                         encoding_strategy,
                         weight_file_name,
                         training=training,
                         learning_rate=learning_rate,
                         num_actors=num_actors,
                         episode_length=episode_length,
                         discount_factor=discount_factor,
                         variance_bias_factor=variance_bias_factor,
                         batch_size=batch_size,
                         clipping_value=clipping_value,
                         num_training_epochs=num_training_epochs,
                         finite_environment=True,
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state: ME_State):
        self.optimizer.zero_grad()

        distribution_params, _, memory = self.network(state.to_cuda_variable())

        if torch.isnan(distribution_params).any():
            raise ValueError('Nan detected')

        distribution = self.create_distribution(distribution_params)

        action = distribution.sample()

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        return action.detach(), memory

    def create_distribution(self, distribution_params):
            alpha = F.softplus(distribution_params[:, 0, :]) + 1
            beta = F.softplus(distribution_params[:, 1, :]) + 1

            return Beta(alpha, beta)

class GeneMutationControl(PPOStrategy):

    """
    Outputs mutation rates for each variable of each individual in the population
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='baseline',
                 learning_rate=5e-6,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=128,
                 batch_size=32,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 fixed_std_deviation=-1,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5):

        num_output_channels = 2

        network = Memory_efficient_network(
            encoding_strategy.num_channels(),
            num_output_channels,
            eliminate_dimension=(0,0,0),
            dim_elimination_max_pooling=dim_elimination_max_pooling,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).cuda()

        super().__init__(network,
                         encoding_strategy,
                         weight_file_name,
                         training=training,
                         learning_rate=learning_rate,
                         num_actors=num_actors,
                         episode_length=episode_length,
                         discount_factor=discount_factor,
                         variance_bias_factor=variance_bias_factor,
                         batch_size=batch_size,
                         clipping_value=clipping_value,
                         num_training_epochs=num_training_epochs,
                         finite_environment=True,
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state: ME_State):
        self.optimizer.zero_grad()

        distribution_params, _, memory = self.network(state.to_cuda_variable())

        if torch.isnan(distribution_params).any():
            raise ValueError('Nan detected')

        distribution = self.create_distribution(distribution_params)

        action = distribution.sample()

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        return action.detach(), memory

    def create_distribution(self, distribution_params):
            alpha = F.softplus(distribution_params[:, 0, :]) + 1
            beta = F.softplus(distribution_params[:, 1, :]) + 1

            return Beta(alpha, beta)


class FitnessShapingControl(PPOStrategy):

    """
    Outputs a fitness shaping factor for each individual of the population
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='baseline',
                 learning_rate=1e-5,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=32,
                 batch_size=16,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 fixed_std_deviation=-1,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5):

        num_output_channels = 1

        network = Memory_efficient_network(
            encoding_strategy.num_channels(),
            num_output_channels,
            eliminate_dimension=(0,1,1),
            dim_elimination_max_pooling=dim_elimination_max_pooling,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).cuda()

        super().__init__(network,
                         encoding_strategy,
                         weight_file_name,
                         training=training,
                         learning_rate=learning_rate,
                         num_actors=num_actors,
                         episode_length=episode_length,
                         discount_factor=discount_factor,
                         variance_bias_factor=variance_bias_factor,
                         batch_size=batch_size,
                         clipping_value=clipping_value,
                         num_training_epochs=num_training_epochs,
                         finite_environment=True,
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state: ME_State):
        self.optimizer.zero_grad()

        distribution_params, _, memory = self.network(state.to_cuda_variable())

        if torch.isnan(distribution_params).any():
            raise ValueError('Nan detected')

        distribution = self.create_distribution(distribution_params)

        action = distribution.sample()

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        return action.detach(), memory

    def create_distribution(self, distribution_params):
        variance = 0.00001 + F.softplus(distribution_params[:, 0, :])
        return Normal(distribution_params[:, 0, :], variance)
