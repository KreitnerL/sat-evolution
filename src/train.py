from utils.training import train_solver, pre_train_solver
from solvers.encoding import ProblemInstanceEncoding
from solvers.solvers import SolverWithIndividualMutationControl
from solvers.solvers import SolverWithGeneMutationControl
from solvers.solvers import SolverWithFitnessShapingCrossover
from solvers.solvers import SolverWithFitnessShapingSelection

import os.path
import sys

encoder = ProblemInstanceEncoding()

print(sys.argv)

solver_arg = sys.argv[1]
outdir = sys.argv[2]
weightsdir = None
if len(sys.argv) > 3:
    weightsdir = sys.argv[3]
start_at = None
if len(sys.argv) > 4:
    start_at = sys.argv[4]


population_size = 100

solverMap = {
    'gene': SolverWithGeneMutationControl(encoder, population_size, num_hidden_layers=3),
    'individual': SolverWithIndividualMutationControl(encoder, population_size, num_hidden_layers=3),
    'crossover': SolverWithFitnessShapingCrossover(encoder, population_size, num_hidden_layers=3),
    'selection': SolverWithFitnessShapingSelection(encoder, population_size, num_hidden_layers=3)
}

solver = solverMap.get(solver_arg, None)
if solver is not None:
    if weightsdir is not None and os.path.isfile(weightsdir + "baseline"):
        print("loading baseline")
        solver.load_weights(weightsdir + "baseline")
    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    if start_at is not None:
        training_stage = start_at.split(':')
        if training_stage[0] == "pre":
            pre_train_solver(solver, outdir, int(training_stage[1]))
            train_solver(solver, outdir)
        elif training_stage[0] == "train":
            train_solver(solver, outdir, int(training_stage[1]))
        raise ValueError("4th argument must be of for pre:INDEX or train:INDEX")
    else:
        pre_train_solver(solver, outdir)
        train_solver(solver, outdir)
