from sat.problem_loader import load_problems

from solvers.encoding import PopulationAndVariablesInInvalidClausesEncoding
from solvers.solvers import *
from timeit import default_timer as timer

import sys
import random
import time

generations = 1000
population_size = 100

def validate(solver, problems, output):
    solver.training = False

    n_problems = len(problems)

    mbf = []

    for i in range(0, generations):
        mbf.append(0)

    for index in range(0, n_problems):
        # TIMING
        start = timer()
        problem = problems[index]
        random.seed(time.time())
        solver.create_population(problem)
        solved = False
        print("Problem: ", problem.get_filename())

        for i in range(0, generations):
            if not solved:
                solver.perform_one_generation((generations - i) / generations)

            if solver.is_solved() and not solved:
                solved = True
                print("Solved in", i, "generations  -  in", (timer()-start), "sec")

            mbf[i] += solver.get_best_score()

        if not solved:
            print("Not solved  -  in", (timer()-start), "sec")

        solver.reset()

    for i in range(0, generations):
        mbf[i] /= n_problems

    with open(output, "w") as f:
        for i in range(0, generations):
            f.write(str(mbf[i]) + '\n')
            f.flush()

encoder = PopulationAndVariablesInInvalidClausesEncoding()

def run_validation(solver, dir):
    problems = load_problems("examples-easy/", "uf20-0", ".cnf", (900,1000))
    validate(solver, problems, dir + "validation-20.txt")
    problems = load_problems("examples-50/", "uf50-0", ".cnf", (900,1000))
    validate(solver, problems, dir + "validation-50.txt")
    problems = load_problems("examples-75/", "uf75-0", ".cnf", (1,100))
    validate(solver, problems, dir + "validation-75.txt")

print(sys.argv)

solver_arg = sys.argv[1]
outdir = sys.argv[2]
weightsdir = sys.argv[3]

solverMap = {
    'gene': SolverWithGeneMutationControl(encoder, population_size, num_hidden_layers=3),
    'individual': SolverWithIndividualMutationControl(encoder, population_size, num_hidden_layers=3),
    'crossover': SolverWithFitnessShapingCrossover(encoder, population_size, num_hidden_layers=3),
    'selection': SolverWithFitnessShapingSelection(encoder, population_size, num_hidden_layers=3),
    'vanila': VanilaSolver(population_size, 0.05)
}

solver = solverMap.get(solver_arg, None)
if solver is not None:
    solver.load_weights(weightsdir + "baseline")
    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    run_validation(solver, outdir)
