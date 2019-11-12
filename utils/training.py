from sat.problem_loader import load_problems

batch_size = 32
pre_training_rounds = 5
training_rounds = 10

def train_problem_set(solver, problems, generations, output, optimize_every):
    solver.training = True

    with open(output, "w") as f:
        for index in range(0, len(problems)):
            problem = problems[index]
            solver.create_population(problem)
            solved = False
            print("Problem: ", problem.get_filename())

            for i in range(0, generations):
                solver.perform_one_generation((generations - i) / generations)
                # print(solver.get_best_score())
                if solver.is_solved() and not solved:
                    solved = True
                    print("Solved in", i, "generations")
                    f.write(str(i)+'\n')
                    break

            if not solved:
                print("Not solved")
                f.write('-1\n')

            solver.reset()
            f.flush()

            if index > 0 and index % optimize_every == 0:
                solver.optimize_network()

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

def pre_train_solver(solver, dir):
    # pre train with easy examples
    for j in range(0, pre_training_rounds):
        problems = load_problems("examples-easy/", "uf20-0", ".cnf", (1,900))
        filename = dir + str(j) + "-pre.txt"
        train_problem_set(solver, problems, 512, filename, 20)
        solver.save_network(dir, j)

def train_solver(solver, dir):
    # alternate easy, medium
    for j in range(1, training_rounds):
        problems = load_problems("examples-easy/", "uf20-0", ".cnf", (1,101))
        filename = dir + str(j) + "-easy.txt"
        train_problem_set(solver, problems, 512, filename, 20)
        solver.save_network(dir, "training-" + str(j))

        problems = load_problems("examples-50/", "uf50-0", ".cnf", (1, 101))
        filename = dir + str(j) + "-50.txt"
        train_problem_set(solver, problems, 1024, filename, 20)
        solver.save_network(dir, "training-" + str(j))
