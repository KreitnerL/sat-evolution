from sat.problem_loader import load_problems
from timeit import default_timer as timer
training_rounds = 10
losses_dir = None

def train_problem_set(solver, problems, generations, output, optimize_every):
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

def train_solver(solver, dir, start_at = 0):
    print("Starting Training:", training_rounds, "rounds:")
    global losses_dir
    losses_dir = dir + "losses"
    # clear content
    open(losses_dir, "w").close()
    # pre train with easy examples
    for j in range(start_at, training_rounds):
        print("Starting training round", j)
        problems = load_problems("DATA/examples-20/", "uf20-", ".cnf", (0,100))
        filename = dir + str(j) + ".txt"
        train_problem_set(solver, problems, 512, filename, 20)
        print("saving network baseline " + str(j) + "...")
        solver.save_network(dir, j)

def save_loss(loss_array: list):
    if not loss_array:
        return
    print("Average Loss:", sum(loss_array)/len(loss_array))
    global losses_dir
    with open(losses_dir, "a") as f:
        f.write("\n".join([str(x) for x in loss_array])+"\n")
        f.flush()
    f.close()