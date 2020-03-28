from sat.problem_loader import load_problems
from timeit import default_timer as timer
from tqdm import tqdm

batch_size = 32
pre_training_rounds = 5
training_rounds = 10
losses_dir = None

def train_problem_set(solver, problems, generations, output, optimize_every):
    solver.training = True

    with open(output, "w") as f:
        for index in range(0, len(problems)):

            # TIMING
            start = timer()

            problem = problems[index]
            solver.create_population(problem)
            solved = False
            print("Problem: ", problem.get_filename())

            for i in tqdm(range(0, generations)):
                solver.perform_one_generation((generations - i) / generations)
                # print(solver.get_best_score())
                if solver.is_solved() and not solved:
                    solved = True
                    # print("Solved in", i, "generations  -  in", (timer()-start), "sec")
                    f.write(str(i)+'\n')
                    break

            if not solved:
                # print("Not solved  -  in", (timer()-start), "sec")
                f.write('-1\n')

            solver.reset()
            f.flush()

            if (index+1) % optimize_every == 0:
                print("\n\noptimizing network...")
                s = timer()
                solver.optimize_network()
                print("optimized network in", timer()-s, "sec\n\n")

    solver.optimize_network()
    solver.clear_experience()

def validate(solver, problems, generations, output):
    solver.training = False

    n_problems = len(problems)

    mbf = []

    for i in range(0, generations):
        mbf.append(0)

    for index in range(0, n_problems):
        problem = problems[index]
        solver.create_population(problem)
        solved = False
        print("Problem: ", problem.get_filename())

        for i in range(0, generations):
            if not solved:
                solver.perform_one_generation((generations - i) / generations)

            if solver.is_solved() and not solved:
                solved = True

            mbf[i] += solver.get_best_score()

        solver.reset()


    for i in range(0, generations):
        mbf[i] /= n_problems

    with open(output, "w") as f:
        for i in range(0, generations):
            f.write(str(mbf[i]) + '\n')
            f.flush()

def pre_train_solver(solver, dir, start_at = 0):
    print("Starting Pre-training:", pre_training_rounds, "rounds:")
    global losses_dir
    losses_dir = dir + "losses-pre"
    # pre train with easy examples
    for j in range(start_at, pre_training_rounds):
        print("Starting pre-training round", j)
        problems = load_problems("DATA/examples-20/", "uf20-", ".cnf", (1,900))
        filename = dir + str(j) + "-pre.txt"
        train_problem_set(solver, problems, 512, filename, 20)
        print("saving network baseline " + str(j) + "...")
        solver.save_network(dir, j)

def train_solver(solver, dir, start_at = 1):
    print("Starting Training:", training_rounds, "rounds:")
    global losses_dir
    losses_dir = dir + "losses-train"
    # alternate easy, medium
    for j in range(start_at, training_rounds):
        print("Starting training round", j)
        problems = load_problems("DATA/examples-easy/", "uf20-0", ".cnf", (1,101))
        filename = dir + str(j) + "-easy.txt"
        train_problem_set(solver, problems, 512, filename, 20)
        print("saving network baseline " + str(j) + "...")
        solver.save_network(dir, "training20-" + str(j))

        problems = load_problems("DATA/examples-50/", "uf50-0", ".cnf", (1, 101))
        filename = dir + str(j) + "-50.txt"
        train_problem_set(solver, problems, 1024, filename, 20)
        print("saving network baseline " + str(j) + "...")
        solver.save_network(dir, "training50-" + str(j))

def save_loss(average_loss, beginn_loss, end_loss):
    print("Average Loss:", average_loss)
    global losses_dir
    with open(losses_dir, "a") as f:
        f.write(str(average_loss) + "\n")
        f.flush()
    f.close()