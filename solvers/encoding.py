"""
This module contains strategies for encoding the state of a generation
for different optimization problems
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


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

class PopulationAndVariablesInInvalidClausesEncoding(EncodingStrategy):
    """
    Encoding strategy to be used for the 3SAT problem.
    """

    def encode(self, population, generations_left):
        """
        :returns: 5D tensor with the following channels/features:\n
                    -genome of each individual\n
                    -fitness of each individual\n
                    -item weights\n
                    -item values\n
                    -problem weight limits\n
                    -remaining number of generations\n
        """
        # Dimensions of each feature: 1 x #Individuals x 2 x #Variables (e.g. 1x100x2x20)

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