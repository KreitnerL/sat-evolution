from sat.problem_loader import load_problems
from solvers.solvers import SolverWithFitnessShapingSelection
from solvers.encoding import PopulationAndVariablesInInvalidClausesEncoding

encoder = PopulationAndVariablesInInvalidClausesEncoding()

solver = SolverWithFitnessShapingSelection(encoder, 5, num_hidden_layers=3)
solver.set_evaluation_function(lambda population: population.evaluate(get_unsatisfied=True))

problems = load_problems("examples-50/", "uf50-0", ".cnf", (1, 101))

solver.create_population(problems[0])

solver.load_weights("presentation/selection/baseline")

generations = 256
with open("presentation/selection-factors.txt", "w+") as log:
    for i in range (0,generations):
        print("generation", str(i))
        solver.perform_one_generation((generations - i) / generations, factors_log=log)