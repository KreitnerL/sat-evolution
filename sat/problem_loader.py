from sat.cnf3 import CNF3
import os

def load_problems(path, prefix, suffix, ids):
    """
    Utility to load many problems to one python list
    """

    problems = []
    start, end = ids
    for i in range(start, end):
        if os.stat(path + prefix + str(i) + suffix).st_size > 0:
            problems.append(CNF3(path + prefix + str(i) + suffix))

    return problems