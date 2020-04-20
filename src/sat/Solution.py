from __future__ import annotations
import random
import numpy as np
import time
from sat.cnf3 import CNF3

class Solution(object):
    """
    Represents one 3SAT solution
    """

    # Initiate a random solution
    def __init__(self, cnf: CNF3, assignments):
        self.cnf = cnf
        # A normalized 2xnum_vars matrix
        self.assignments = assignments

    @staticmethod
    def random(cnf):
        """
        returns a 2xNum_vars matrix
        """

        # random.seed(time.time())

        assignments = np.zeros((2,cnf.num_variables), np.int8)
        for i in range(0, cnf.num_variables):
            assignments[random.getrandbits(1)][i] = 1

        return Solution(cnf, assignments)


    def evaluate(self, get_unsatisfied=False):
        self.score, self.τ_satisfied_clauses, self.make_values, self.break_values = self.cnf.evaluate(self.assignments, get_additional_properties=get_unsatisfied)
        self.has_unsatisfied = get_unsatisfied

    def mutate(self, probability, per_gene=False):
        """
        Performs mutation (flip value) using the supplied probabilty. \n
        If per_gene is set, assumes probability is an array that contains mutation rates for each variable
        """
        for i in range(0, self.cnf.num_variables):
            if random.uniform(0,1) < (probability[i] if per_gene else probability):
                self.assignments[:,i] = 1-self.assignments[:,i]

    def crossover(self, other) -> Solution:
        """
        Perform crossover of two solutions
        """
        # random.seed(time.time())

        assignments = np.zeros((2,self.cnf.num_variables), np.int8)

        for i in range(0, self.cnf.num_variables):
            if bool(random.getrandbits(1)):
                assignments[:,i] = self.assignments[:,i]
            else:
                assignments[:,i] = other.assignments[:,i]

        return Solution(self.cnf, assignments)

    def local_search(self, max_variables):
        """
        Perform greedy local search up to max_variables times, try flipping one variable at a time and 
        persist the change that has increased fitness the most
        """
        assignments = self.assignments.copy()

        best_var = None
        best_improvement = 0

        for _ in range (0, max_variables):
            for var in range(0, self.cnf.num_variables):
                self.assignments[:,var] = 1-self.assignments[:,var]
                score, _, __, ___ = self.cnf.evaluate(assignments)
                improvement = score - self.get_score()
                if improvement > 0 and improvement > best_improvement:
                    best_improvement = improvement
                    best_var = var

                self.assignments[:,var] = 1-self.assignments[:,var]

            if best_improvement > 0:
                self.assignments[:,best_var] = 1-self.assignments[:,best_var]

        self.assignments = assignments

    def get_assignments(self) -> np.ndarray:
        return self.assignments

    def get_inverse_assignments(self) -> np.ndarray:
        return 1 - self.assignments.copy()

    def get_score(self) -> int:
        return self.score

    def get_make_values(self) -> np.ndarray:
        return self.make_values

    def get_break_values(self) -> np.ndarray:
        return self.break_values

    def get_satisfied_clauses(self) -> np.ndarray:
        return self.τ_satisfied_clauses