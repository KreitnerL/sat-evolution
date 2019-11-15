from sat.Solution import Solution

import random


class Population(object):
    """
    Represent a population of SAT3 solutions
    """

    def __init__(self, solutions, cnf):
        self.cnf = cnf
        self.solutions = solutions
        self.size = len(solutions)

    @staticmethod
    def random(cnf, size):
        solutions = []

        for _ in range(size):
            solutions.append(Solution.random(cnf))

        return Population(solutions, cnf)

    def get_size(self):
        return self.size

    def get_solutions(self):
        return self.solutions

    def is_solved(self):
        return self.solutions[0].get_score() == self.cnf.num_clauses

    def get_best_solution(self):
        return sorted(self.solutions, key=lambda s: s.get_score(), reverse=True)[0]

    def evaluate(self, get_unsatisfied=False):
        for solution in self.solutions:
            solution.evaluate(get_unsatisfied=get_unsatisfied)

    def mutate(self, probabilities, per_gene=False):
        index = 0
        for solution in self.solutions:
            solution.mutate(probabilities[index], per_gene=per_gene)
            if len(probabilities) > 1:
                index += 1

    def modify_fitness(self, factors):
        for i in range(self.get_size()):
            self.solutions[i].score *= factors[i]

    def crossover(self, n_best=0):
        size = self.get_size()

        if n_best > len(self.solutions) or n_best < 2:
            raise Exception

        self.solutions = sorted(self.solutions, key=lambda s: s.get_score(), reverse=True)[0:n_best]

        for _ in range(0, size - n_best):
            parent1, parent2 = Population._pick_two(self.solutions)
            child1 = parent1.crossover(parent2)
            child2 = parent1.crossover(parent2)
            self.solutions.append(child1)
            self.solutions.append(child2)

    def restart(self):
        for i in range(self.get_size()):
            self.solutions[i] = Solution.random(self.cnf)

    def selection(self, size):
        self.solutions = self.solutions[0:size]

    def local_search(self, num_solutions, max_variables):
        for i in range(0, num_solutions):
            self.solutions[i].local_search(max_variables)

    @staticmethod
    def _pick_two(solutions):
        if len(solutions) == 2:
            return solutions

        one = random.choice(solutions)
        two = random.choice(solutions)

        if one != two:
            return one, two
        else:
            return Population._pick_two(solutions)