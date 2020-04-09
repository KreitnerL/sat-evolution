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

    NUM_DIMENSIONS=4
    """
    Improved encoding strategy to be used for the 3SAT problem.
    """
    def encode(self, population: Population, generations_left, memory: List[torch.Tensor] = None) -> ME_State:
        """
        :returns: ME_State with all relevant features in the form of [B,C,P,E,G,G].\n
        B = Batch size
        C = Channel size
        P = Population size
        E = Equation size (#clauses)
        G = Genome size (#variables)
        """
        P = population.get_size()
        G = population.cnf.num_variables
        E = population.cnf.num_clauses
        # Problem instance (2x)1xGxE
        problem  = torch.tensor(population.cnf.mat).float().view(1,2,1,E,G,1)

        # Solution of each individual (genome) (2x)PxGx1
        population_data = torch.tensor([solution.get_assignments() for solution in population.get_solutions()]).float().permute(1,0,2).view(1,2,P,1,G,1)

        # Participation in clauses 1xGx1
        variable_participation = torch.tensor(population.cnf.get_participation() / population.cnf.num_clauses).float().view(1,1,1,1,G,1)

        # Make value per genome PxGx1
        make_values = torch.tensor([solution.get_make_values() / population.cnf.num_clauses for solution in population.get_solutions()]).float().view(1,1,P,1,G,1)

        # Break value per genome PxGx1
        break_values = torch.tensor([solution.get_break_values() / population.cnf.num_clauses for solution in population.get_solutions()]).float().view(1,1,P,1,G,1)

        # Satisfied clauses Px1xE
        τ_satisfied_clauses = torch.tensor([solution.get_satisfied_clauses() for solution in population.get_solutions()]).float().view(1,1,P,E,1,1)

        # Fitness of each individual Px1x1
        population_fitness = torch.tensor([solution.get_score() for solution in population.get_solutions()]).float().view(1,1,P,1,1,1)

        # Number of clauses 1x1x1
        num_clauses = torch.tensor([population.cnf.num_clauses]).float().view(1,1,1,1,1,1)

        # Number of variables 1x1x1
        num_vars = torch.tensor([population.cnf.num_variables]).float().view(1,1,1,1,1,1)

        # Generations_left 1x1x1
        generations_left = torch.tensor([generations_left]).float().view(1,1,1,1,1,1)

        # Initialize memory with 0
        if not memory and self.num_channels()[1]:
            memory = ME_State([torch.zeros(1, channels, P if p else 1, E if e else 1, G if g else 1, G if g2 else 1).cuda() for (p,e,g,g2), channels in self.num_channels()[1].items()])

        return ME_State([problem,
                        population_data,
                        make_values,
                        break_values,
                        τ_satisfied_clauses,
                        population_fitness,
                        variable_participation,
                        num_clauses, 
                        num_vars, 
                        generations_left],
                        memory)
    
    def num_channels(self) -> tuple:
        """
        Returns a Tuple with: \n
        1) Dict input_stream_code to number of channels.\n
        2) Maps memory_stream_code to number of channels.\n
        3) List of practical feature codes
        4) List of theoretical feature codes
        """
        P = G = E = 1
        # Dict of all feature dimensions and their respective channels
        features = Counter({
            (0,E,G,0): 2,
            (P,0,G,0): 4,
            (P,E,0,0): 1,
            (0,0,G,0): 1,
            (P,0,0,0): 1,
            (0,0,0,0): 3
        })
        # Dict of all memory feature dimensions and their respective channels.
        memory_dim = Counter({
            (P,0,G,0): 5,
            (P,E,0,0): 5,
            (P,0,0,0): 5,
            (0,E,G,0): 5,
            (0,0,G,G): 5
        })
        # List of all features that change dynamically while searching for a solution
        practical_features = [
            (P,0,G,0),
            (P,E,0,0),
            (P,0,0,0),
            (0,E,G,0),
            (0,0,0,0)
        ]
        # List of all features that are used for long term inferences
        theoretical_features = [
            (0,E,G,0),
            (0,0,G,G),
            (0,0,G,0),
            (0,0,0,0)
        ]
        # results of the theoretical reasoining that are used for practical reasoning
        report = [
            (0,E,G,0)
        ]
        return features+memory_dim, memory_dim, practical_features, theoretical_features, report

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