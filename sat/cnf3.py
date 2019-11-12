import numpy as np

class CNF3(object):
    """
    Represents a single SAT3 problem
    """
    num_variables = 0
    num_clauses = 0

    def __init__(self, path):
        self.filename = path
        self.participation = None

        with open(path) as f:
            index = 0
            for line in f:
                if line[0] == 'c':
                    continue

                if line[0] == 'p':
                    split = line.strip().replace("  ", " ").split(" ")
                    self.num_variables, self.num_clauses = int(split[2]), int(split[3])
                    self.mat = np.zeros((self.num_clauses, self.num_variables), np.int8)
                    continue

                split = line.strip().replace("  ", " ").split(" ")
                for var_str in split:
                    if var_str == '%':
                        return

                    var = int(var_str)
                    if var == 0:
                        break

                    if var > 0:
                        var -= 1  # 1 indexing
                        self.mat[index][var] = 1
                    else:
                        var += 1  # 1 indexing
                        self.mat[index][-var] = -1

                index += 1

    def evaluate(self, solution, get_unsatisfied=False):
        """
        Evaluate one solution, return number of satisfied clauses.
        If get_unsatisfied is set, return a numpy array where element i is the number of unsatisfied clauses that
        contain variable i
        """

        result = np.multiply(self.mat, -1 * solution)

        # Count satisfied
        result_satisfied = result.copy()
        result_satisfied[result_satisfied > 0] = 0
        satisfied = np.count_nonzero(result_satisfied.sum(1))

        # Get variables in unsatisfied clauses
        unsatisfied = None
        if get_unsatisfied:
            result[result < 0] = 0
            unsatisfied = result.sum(1)
            unsatisfied[unsatisfied < 3] = 0
            unsatisfied = unsatisfied / 3

        return satisfied, unsatisfied

    # for each variable, number of clauses that contain it
    def get_participation(self):
        if self.participation is None:
            mat = self.mat.copy()
            mat[mat == -1] = 1
            self.participation = mat.sum(0)

        return self.participation


    def get_filename(self):
        return self.filename