"""
This module contains strategies for encoding the state of a generation
for different optimization problems
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sat.population import Population
from neural_networks.memory_efficient_state import ME_State
from collections import Counter
from typing import List


class EncodingStrategy(ABC):
    """
    An EncodingStrategy encodes the state of a population on a specific problem instance
    """

    def __init__(self, problem=None, num_problem_dimensions=None):
        self.problem = problem
        self.num_problem_dimensions = num_problem_dimensions

    @abstractmethod
    def encode(self, population, generations_left):
        """
        :parameter population: The population to encode
        :parameter generations_left: Encoding of the remaining number of generations
        :returns: The encoded state of a population
        """
        pass

    @abstractmethod
    def num_channels(self):
        pass


class ProblemInstanceEncoding(EncodingStrategy):
    """
    Improved encoding strategy to be used for the 3SAT problem.
    """
    def encode(self, population: Population, generations_left, memory: List[torch.Tensor] = None) -> ME_State:
        """
        :returns: ME_State with following features:\n
            - problem instance (2x)1xGxE
            - genome of each individual (2x)PxGx1
            - fitness of each individual Px1x1
            - Participation of each variable in clauses 1xGx1
            - Participation of each variable in unsatistfied clauses 1xGx1
            - Satisfied clauses Px1xE
            - Number of clauses 1x1x1
            - Number of variables 1x1x1
            - Generations left 1x1x1
            - memory of the last step
        """
        P = population.size
        G = population.cnf.num_variables
        E = population.cnf.num_clauses
        # Feature 0: Problem instance (2x)1xGxE
        problem  = torch.tensor(population.cnf.mat).float().permute(0,2,1).view(1,2,1,G,E)

        # Feature 1: Solution of each individual (genome) (2x)PxGx1
        population_data = torch.tensor([solution.get_assignments() for solution in population.get_solutions()]).float().permute(1,0,2).view(1,2,P,G,1)

        # Feature 2 : Participation in clauses 1xGx1
        variable_participation = torch.tensor(population.cnf.get_participation() / population.cnf.num_clauses).float().view(1,1,1,G,1)

        # Feature 3 : Participation in unsatistfied clauses PxGx1
        variable_participation_in_unsatisfied = torch.tensor([solution.get_unsatisfied() / population.cnf.num_clauses for solution in population.get_solutions()]).float().view(1,1,P,G,1)

        # Feature 4 : Satisfied clauses Px1xE
        satisfied_clauses = torch.tensor([solution.get_satisfied_clauses() for solution in population.get_solutions()]).float().view(1,1,P,1,E)

        # Feature 5: Fitness of each individual Px1x1
        population_fitness = torch.tensor([solution.get_score() for solution in population.get_solutions()]).float().view(1,1,P,1,1)

        # Feature 6 : Number of clauses 1x1x1
        num_clauses = torch.tensor([population.cnf.num_clauses]).float().view(1,1,1,1,1)

        # Feature 7 : Number of variables 1x1x1
        num_vars = torch.tensor([population.cnf.num_variables]).float().view(1,1,1,1,1)

        # Feature 8: Generations_left 1x1x1
        generations_left = torch.tensor([generations_left]).float().view(1,1,1,1,1)

        # Initialize memory with values 0.5
        if not memory and self.num_channels()[1]:
            memory = ME_State([torch.zeros(1,channels, P if p else 1, G if g else 1, E if e else 1).cuda() for (p,g,e), channels in self.num_channels()[1].items()])

        return ME_State([problem,
                        population_data,
                        variable_participation_in_unsatisfied,
                        satisfied_clauses,
                        population_fitness,
                        variable_participation,
                        num_clauses, 
                        num_vars, 
                        generations_left],
                        memory)
    
    def num_channels(self):
        """
        Returns a tuple of dictionaries. 1) Maps input_stream code to number of channels. 2) Maps memory_stream code to number of channels.
        """
        features = Counter({
            (0,1,1): 2,
            (1,1,0): 3,
            (1,0,1): 1,
            (0,1,0): 1,
            (1,0,0): 1,
            (0,0,0): 3
        })
        memory_dim = Counter({
            (1,1,0): 5,
            (1,0,1): 5,
            (1,0,0): 5
        })
        return features+memory_dim, memory_dim

class PopulationAndVariablesInInvalidClausesEncoding(EncodingStrategy):
    """
    Encoding strategy to be used for the 3SAT problem.
    """

    def encode(self, population, generations_left):
        """
        :returns: 4D tensor with the following channels/features:\n
                    -genome of each individual
                    -fitness of each individual
                    -item weights
                    -item values
                    -problem weight limits
                    -remaining number of generations
        """
        # TODO
        # Dimensions of each feature: 1 x #Individuals x (#Variables*2) (e.g. 1x100x40)

        # Feature 0 : Solution of each individual
        population_data = torch.tensor(
            [solution.get_assignments().reshape(solution.get_assignments().size) for solution in population.get_solutions()]
            ).unsqueeze(0).float()

        # Feature 1 : Inverse solution of each individual
        array = []
        for solution in population.get_solutions():
            assignments = solution.get_inverse_assignments()
            array.append(assignments.reshape(assignments.size))

        population_inverse = torch.tensor(array).unsqueeze(0).float()

        # Feature 2 : Fitness of each individual
        population_fitness = torch.tensor([[solution.get_score()] for solution in population.get_solutions()])
        population_fitness = population_fitness.expand_as(population_data).float()

        # Feature 3 : Participation in clauses:
        variable_participation = torch.tensor(
            [np.tile(population.cnf.get_participation(),2) / population.cnf.num_clauses for solution in population.get_solutions()]
            ).unsqueeze(0).float()

        # Feature 4 : Participation in unsatistfied clauses
        in_unsatisfied = torch.tensor(
            [np.tile(solution.get_unsatisfied(),2) / population.cnf.num_clauses for solution in population.get_solutions()]
            ).unsqueeze(0).float()

        # Feature 5 : Number of clauses
        num_clauses = torch.tensor(population.cnf.num_clauses).float()
        num_clauses = num_clauses.expand_as(population_data)

        # Feature 6 : Number of variables
        num_vars = torch.tensor(population.cnf.num_variables).float()
        num_vars = num_vars.expand_as(population_data)

        # Feature 7: Generations_left
        generations_left = torch.tensor(generations_left).float()
        generations_left = generations_left.expand_as(population_data)

        # Create #Features x #individuals x individual_length tensor
        state = torch.cat([population_data,
                           population_inverse,
                           in_unsatisfied,
                           population_fitness,
                           num_clauses,
                           num_vars,
                           variable_participation,
                           generations_left], 0)


        # Add batch_dimension of size 1
        state = state.unsqueeze(0)

        return state.float()

    def num_channels(self):
        return 8