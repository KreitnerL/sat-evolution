from sat.problem_loader import load_problems
from timeit import default_timer as timer
from solvers.encoding import ProblemInstanceEncoding
from solvers.solvers import *
from utils.plotter import set_loss_directory
import os
training_rounds = 10

def solve_problem_set(solver, problems, generations, output, optimize_every):
    solver.training = True
    # clear content
    open(output, "w").close()
    for index in range(0, len(problems)):
        with open(output, "a") as f:
            # TIMING
            start = timer()

            problem = problems[index]
            solver.create_population(problem)
            solved = False
            print("Problem: ", problem.get_filename())
            for i in range(generations):
                if not solved:
                    solver.perform_one_generation((generations - i) / generations)
                if solver.is_solved() and not solved:
                    solved = True
                    print("Solved in", i, "generations  -  in", (timer()-start), "sec")
                    f.write(str(i)+'\n')
                    break

            if not solved:
                print("Not solved  -  in", (timer()-start), "sec")
                f.write('-1\n')

            solver.reset()
            f.flush()

            if (index+1) % optimize_every == 0:
                print("\n\noptimizing network...")
                s = timer()
                solver.optimize_network()
                print("optimized network in", timer()-s, "sec\n\n")
    if optimize_every != float('inf'):
        solver.optimize_network()
        solver.clear_experience()

def train_solver(solver, dir, start_at = 0):
    print("Starting Training:", training_rounds, "rounds:")
    set_loss_directory(dir + "losses")
    for j in range(start_at, training_rounds):
        print("Starting training round", j)
        problems = load_problems("DATA/examples-20/", "uf20-", ".cnf", (0,100))
        filename = dir + str(j) + ".txt"
        solve_problem_set(solver, problems, 512, filename, 20)
        print("saving network baseline " + str(j) + "...")
        solver.save_network(dir, j)

def validate_solver(solver, dir):
    max_generations = 512
    problems = load_problems("DATA/examples-20/", "uf20-", ".cnf", (900,1000))
    solve_problem_set(solver, problems, max_generations, dir + "validation-20.txt", float('inf'))
    problems = load_problems("DATA/examples-easy/", "uf20-0", ".cnf", (900,1000))
    solve_problem_set(solver, problems, max_generations, dir + "validation-easy.txt", float('inf'))

population_size = 100
solverMap = {
    'gene': SolverWithGeneMutationControl,
    'individual': SolverWithIndividualMutationControl,
    'crossover': SolverWithFitnessShapingCrossover,
    'selection': SolverWithFitnessShapingSelection,
    'vanilla': VanillaSolver
}
encoder = ProblemInstanceEncoding()

def start(training, solver_arg, outdir, weightsdir, start_at):
    hyperparamter = encoder.get_hyperparamter()
    if solver_arg == 'vanilla':
        solver = VanillaSolver(population_size, 0.05)
    else:
        solver = solverMap.get(solver_arg, None)(encoder, population_size, num_neurons = hyperparamter["NUM_NEURONS"], num_hidden_layers=hyperparamter["NUM_HIDDEN_LAYERS"], learning_rate=hyperparamter["LEARNING_RATE"])
    if solver is not None:
        if weightsdir is not None and os.path.isfile(weightsdir + "baseline"):
            print("loading baseline")
            solver.load_weights(weightsdir + "baseline")
        solver.set_evaluation_function(lambda population : population.evaluate(get_unsatisfied=True))
        if training:
            if start_at is not None:
                    train_solver(solver, outdir, int(start_at))
            else:
                train_solver(solver, outdir)
        else:
            validate_solver(solver, outdir)