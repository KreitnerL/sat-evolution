from sat.problem_loader import load_problems
from solvers.solvers import SolverWithGeneMutationControl
from solvers.encoding import ProblemInstanceEncoding

encoder = ProblemInstanceEncoding()

solver = SolverWithGeneMutationControl(encoder, 100, num_hidden_layers=3)
solver.set_evaluation_function(lambda population: population.evaluate(get_unsatisfied=True))

problems = load_problems("examples-50/", "uf50-0", ".cnf", (1, 2))

solver.create_population(problems[0])

solver.load_weights("presentation/gene-mutation/baseline")

generations = 256
with open("presentation/gene-mutation-rates.txt", "w+") as log:
    for i in range (0,generations):
        print("generation", str(i))
        solver.perform_one_generation((generations - i) / generations, mutation_rates_log=log)