import numpy as np
from typing import Tuple
from io import TextIOWrapper

class CNF3(object):
    """
    Represents a single SAT3 problem
    """
    
    def __init__(self, path):
        self.filename = path
        self.participation: np.ndarray = None
        # A 2xExG Matrix storing the 3sat problem with E = Number Clauses, G = Number Variables, and 2 seperate channels for true/false
        self.mat: np.ndarray = None
        self.num_variables = 0
        self.num_clauses = 0

        with open(path) as f:
            self.parseInputFile(f)
        self.participation = np.bitwise_or(self.mat[0], self.mat[1])

    def parseInputFile(self, f: TextIOWrapper):
        """
        Parses the given file and sets self.mat accordingly.
        :param f: file containing a 3SAT problem
        """
        clause_index = 0
        for line in f:
            if line[0] == 'c':
                continue

            if line[0] == 'p':
                split = line.strip().replace("  ", " ").split(" ")
                self.num_variables, self.num_clauses = int(split[2]), int(split[3])
                self.mat = np.zeros((2, self.num_clauses, self.num_variables), np.int8)
                continue

            split = line.strip().replace("  ", " ").split(" ")
            for var_str in split:
                if var_str == '%':
                    return

                var_index = int(var_str)
                if var_index == 0:
                    break

                if var_index > 0:
                    self.mat[0][clause_index][var_index-1] = 1
                else:
                    self.mat[1][clause_index][-var_index-1] = 1

            clause_index += 1

    def evaluate(self, solution: np.ndarray, get_additional_properties=False) -> Tuple[np.ndarray]:
        """
        Evaluate the given assignment and returns 
        - satisfied: number of satisfied clauses
        - τ_satisfied_clauses: number of true literals per clause

        Following will only be calculated if get_additional_properties=True
        - make_value: make_value per genome (number of clauses that turn true if var is flipped)
        - break_value: break_value per genome (number of clauses that turn false if var is flipped)
        """
        # calculate clause assignment
        result = np.array([np.bitwise_and(self.mat[0], solution[0]), np.bitwise_and(self.mat[1], solution[1])])
        # flatten to a ExG Matrix. 0 = false, 1 = true
        result2D = np.bitwise_or(result[0], result[1])

        τ_satisfied_clauses = result2D.sum(1)
        satisfied = np.count_nonzero(τ_satisfied_clauses)

        make_value = None
        break_value = None
        if get_additional_properties:
            make_value = np.array([self.participation[clause] for clause in range(self.num_clauses) if τ_satisfied_clauses[clause] == 0]).sum(0)
            if not isinstance(make_value, list):
                make_value = np.zeros(self.num_variables)
            break_value = np.array([self.participation[clause] for clause in range(self.num_clauses) if τ_satisfied_clauses[clause] == 1]).sum(0)
            if not isinstance(break_value, list):
                make_value = np.zeros(self.num_variables)
            
            
        return satisfied, τ_satisfied_clauses, make_value, break_value

    def get_participation(self) -> np.ndarray:
        """
        returns for each variable, number of clauses that contain it
        """
        return self.participation.sum(0)


    def get_filename(self) -> str:
        return self.filename