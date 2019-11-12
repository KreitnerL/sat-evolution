from utils.training import train_solver, pre_train_solver
from solvers.encoding import PopulationAndVariablesInInvalidClausesEncoding
from solvers.solvers import SolverWithIndividualMutationControl
from solvers.solvers import SolverWithGeneMutationControl
from solvers.solvers import SolverWithFitnessShapingCrossover
from solvers.solvers import SolverWithFitnessShapingSelection

import os.path
import sys

encoder = PopulationAndVariablesInInvalidClausesEncoding()

print(sys.argv)

solver_arg = sys.argv[1]
outdir = sys.argv[2]


population_size = 100

if solver_arg == 'gene':
    solver = SolverWithGeneMutationControl(encoder, population_size, num_hidden_layers=3)
    if os.path.isfile(outdir + "baseline"):
        print("loading baseline")
        solver.load_weights(outdir + "baseline")

    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    pre_train_solver(solver, outdir)
    train_solver(solver, outdir)

if solver_arg == 'individual':
    solver = SolverWithIndividualMutationControl(encoder, population_size, num_hidden_layers=3)
    if os.path.isfile(outdir + "baseline"):
        print("loading baseline")
        solver.load_weights(outdir + "baseline")

    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    pre_train_solver(solver, outdir)
    train_solver(solver, outdir)

if solver_arg == 'crossover':
    solver = SolverWithFitnessShapingCrossover(encoder, population_size, num_hidden_layers=3)
    if os.path.isfile(outdir + "baseline"):
        print("loading baseline")
        solver.load_weights(outdir + "baseline")

    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    pre_train_solver(solver, outdir)
    train_solver(solver, outdir)

if solver_arg == 'selection':
    solver = SolverWithFitnessShapingSelection(encoder, population_size, num_hidden_layers=3)
    if os.path.isfile(outdir + "baseline"):
        print("loading baseline")
        solver.load_weights(outdir + "baseline")

    solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
    pre_train_solver(solver, outdir)
    train_solver(solver, outdir)
