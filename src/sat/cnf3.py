import numpy as np

class CNF3(object):
    """
    Represents a single SAT3 problem
    """
    num_variables = 0
    num_clauses = 0

    def __init__(self, path):
        self.filename = path
        self.participation: np.ndarray = None

        with open(path) as f:
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

    def evaluate(self, solution, get_unsatisfied=False):
        """
        Evaluate one solution, return number of satisfied clauses.
        If get_unsatisfied is set, return a numpy array where element i is the number of unsatisfied clauses that
        contain variable i
        """

        #calculate clause assignment
        result = np.array([np.bitwise_and(self.mat[0], solution[0]), np.bitwise_and(self.mat[1], solution[1])])
        # flatten to a 2D Matrix
        result2D = np.bitwise_or(result[0], result[1])

        # Count satisfied
        satisfied_clauses = np.bitwise_or.reduce(result2D,1)
        satisfied = np.count_nonzero(satisfied_clauses)

        # Get variables in unsatisfied clauses
        unsatisfied = None
        unsatisfied_count_vector = np.zeros(self.num_variables, np.int8)
        if get_unsatisfied:
            unsatisfied = 1-satisfied_clauses
            mat2d = np.bitwise_or(self.mat[0], self.mat[1])
            for i in range(unsatisfied.size):
                mat2d[i]*=unsatisfied[i]
            unsatisfied_count_vector = np.bitwise_or.reduce(mat2d,0)
            
        return satisfied, unsatisfied_count_vector, satisfied_clauses

    def get_participation(self):
        """
        returns for each variable, number of clauses that contain it
        """
        if self.participation is None:
            self.participation = np.bitwise_or(self.mat[0], self.mat[1]).sum(0)

        return self.participation


    def get_filename(self):
        return self.filename