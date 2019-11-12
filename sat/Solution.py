import random
import numpy as np
import time

class Solution(object):
    """
    Represents one 3SAT solution
    """

    # Initiate a random solution
    def __init__(self, cnf, assignments):
        self.cnf = cnf
        self.assignments = assignments
        random.seed(time.time())

    @staticmethod
    def random(cnf):

        random.seed(time.time())

        assignments = np.zeros((cnf.num_variables), np.int8)
        for i in range(0, cnf.num_variables):
            if bool(random.getrandbits(1)):
                assignments[i] = 1
            else:
                assignments[i] = -1

        return Solution(cnf, assignments)


    def evaluate(self, get_unsatisfied=False):
        self.score, self.variables_in_unsat = self.cnf.evaluate(self.assignments, get_unsatisfied=get_unsatisfied)
        self.has_unsatisfied = get_unsatisfied

    # Perform mutation using the supplied probabilty
    # If per_gene is set, assumes probability is an array that contains mutation rates for each variable
    def mutate(self, probability, per_gene=False):
        if per_gene:
            for i in range(0, self.cnf.num_variables):
                if probability[i]:
                    self.assignments[i] *= -1
        else:
            for i in range(0, self.cnf.num_variables):
                if random.uniform(0,1) < probability:
                    self.assignments[i] *= -1

    # Perform crossover of two solutions
    def crossover(self, other):
        random.seed(time.time())

        assignments = np.zeros((self.cnf.num_variables), np.int8)

        for i in range(0, self.cnf.num_variables):
            if bool(random.getrandbits(1)):
                assignments[i] = self.assignments[i]
            else:
                assignments[i] = other.assignments[i]

        return Solution(self.cnf, assignments)

    # Perform greedy local search
    # up to max_variables times, try flipping one variable at a time and persist the change that has increased fitness
    # the most
    def local_search(self, max_variables):
        assignments = self.assignments.copy()

        best_var = None
        best_improvement = 0

        for i in range (0, max_variables):
            for var in range(0, self.cnf.num_variables):
                assignments[var] = -1 * assignments[var]
                score, _ = self.cnf.evaluate(assignments)
                improvement = score - self.get_score()
                if improvement > 0 and improvement > best_improvement:
                    best_improvement = improvement
                    best_var = var

                assignments[var] = -1 * assignments[var]

            if best_improvement > 0:
                assignments[best_var] = -1 * assignments[best_var]

        self.assignments = assignments

    def get_assignments(self):
        return self.assignments

    def get_inverse_assignments(self):
        inverse = self.assignments.copy()
        inverse -= 1
        inverse *= -1
        return inverse

    def get_score(self):
        return self.score

    # A vector that encodes for each variable, if it participates in an unsat. clause
    # 1: Yes, 0: No
    def get_unsatisfied(self):
        if not self.has_unsatisfied:
            raise Exception("Unsatisfied not evaluated")
        return self.assignments