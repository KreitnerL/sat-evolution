from strategies.startegies import *
from abc import abstractmethod
from sat.population import Population
from math import log

class SatSolver(object):
    '''
     Has specific network output, configurable input
     '''

    def __init__(self, input_encoder, population_size, strategy: PPOStrategy, num_hidden_layers=1, satisfied_reward_factor=2):
        self.input_encoder = input_encoder
        self.population_size = population_size
        self.satisfied_reward_factor = satisfied_reward_factor

        self.population = None
        self.problem = None
        self.evaluation_function = lambda population : population.evaluate()

        self.best_score = 0
        self.strategy = strategy

        self.training = True

    @abstractmethod
    def create_population(self, problem, size):
        pass

    @abstractmethod
    def perform_one_generation(self, generations_left):
        pass

    def get_best_score(self):
        return self.best_score

    # Replace default evaluation function in case needed to extract features
    def set_evaluation_function(self, evaluation_function):
        self.evaluation_function = evaluation_function

    def optimize_network(self):
        self.strategy.optimize_model()

    def save_network(self, dir, iteration):
        self.strategy.store_weights(dir, iteration)

    def load_weights(self, filename):
        self.strategy.load_weights(filename)

    def reset(self):
        self.population = None
        self.problem = None
        self.best_score = 0

    def clear_experience(self):
        self.strategy.episode_memory = []
        self.strategy.actor_experience_store = []

    def is_solved(self):
        return self.best_score == self.problem.num_clauses

    def evaluate_and_reward(self, generations_left):
        self.evaluation_function(self.population)
        score = self.population.get_best_solution().get_score()

        if self.best_score == 0:
            self.best_score = score

        # if solved, increase reward
        if score == self.problem.num_clauses:
            reward = self.calc_reward(self.best_score, score * self.satisfied_reward_factor)
        elif score >= self.best_score:
            reward = self.calc_reward(self.best_score, score)
        else:
            reward = self.calc_reward(self.best_score, self.best_score)

        state = self.input_encoder.encode(self.population, generations_left)

        if self.training:
            self.strategy.reward(reward, state)

        if score > self.best_score:
            self.best_score = score

    def calc_reward(self, old_best_fitness, new_best_fitness):
        """
        Calculate the reward for an action, based on the change in best fitness
        """
        if old_best_fitness == 0:
            if new_best_fitness == 0:
                return 0
            else:
                return 100 * log(new_best_fitness, 10)
        if new_best_fitness == 0:
            return - 100 * abs(log(1 + old_best_fitness, 10))
        return 100 * log(new_best_fitness / old_best_fitness, 10)


class SolverWithIndividualMutationControl(SatSolver):
    '''
     Solver that uses the network to control individual mutation rates
     '''

    def __init__(self, input_encoder, population_size, num_hidden_layers=1, satisfied_reward_factor=2):
        strategy = IndividualMutationControl(input_encoder, "", 4, training=True, num_hidden_layers=num_hidden_layers)
        super().__init__(input_encoder, population_size, strategy, num_hidden_layers=num_hidden_layers, satisfied_reward_factor=satisfied_reward_factor)

    def create_population(self, problem):
        self.population = Population.random(problem, self.population_size)
        self.evaluation_function(self.population)
        self.problem = problem

    def perform_one_generation(self, generations_left, mutation_rates_log=None):
        self.population.local_search(5, 1)
        self.population.crossover(n_best=int(self.population_size / 2))
        self.population.selection(self.population_size)
        self.evaluation_function(self.population)

        state = self.input_encoder.encode(self.population, generations_left)
        mutation_rates = self.strategy.select_action(state).data.cpu().numpy()[0]

        # Log mutation rates to file
        if mutation_rates_log is not None:
            sorted = []
            for rate in mutation_rates:
                sorted.append(rate)

            sorted.sort()
            for rate in sorted:
                mutation_rates_log.write(str(rate) + ' ')

            mutation_rates_log.write('\n')
            mutation_rates_log.flush()

        self.population.mutate(mutation_rates)

        self.evaluate_and_reward(generations_left)

class SolverWithGeneMutationControl(SatSolver):
    '''
     Solver that uses the network to control individual mutation rates, gene by gene.
     The network outputs a mutation rate for each variable of each solution.
     '''

    def __init__(self, input_encoder, population_size, num_hidden_layers=1, satisfied_reward_factor=2):
        strategy = GeneMutationControl(input_encoder, "", 4, training=True, num_hidden_layers=num_hidden_layers)
        super().__init__(input_encoder, population_size, strategy, num_hidden_layers=num_hidden_layers)

    def create_population(self, problem):
        self.population = Population.random(problem, self.population_size)
        self.evaluation_function(self.population)
        self.problem = problem

    def perform_one_generation(self, generations_left, mutation_rates_log=None):
        self.population.local_search(1, 1)
        self.population.crossover(n_best=int(self.population_size / 2))
        self.population.selection(self.population_size)
        self.evaluation_function(self.population)

        state = self.input_encoder.encode(self.population, generations_left)
        mutation_rates = self.strategy.select_action(state).data.cpu().numpy()[0]

        # Log mutation rates to file
        if mutation_rates_log is not None:
            sorted = []
            for rates in mutation_rates:
                sorted.append(np.mean(rates))

            sorted.sort()
            for rate in sorted:
                mutation_rates_log.write(str(rate) + ' ')

            mutation_rates_log.write('\n')
            mutation_rates_log.flush()

        self.population.mutate(mutation_rates, per_gene=True)

        self.evaluate_and_reward(generations_left)

class SolverWithFitnessShapingCrossover(SatSolver):
    '''
     Solver that uses the network to control crossover.
     The network outputs factors for each individuals, which are multiplied with the individual fitness.
     This is done before the crossover step so the network can choose which individuals are available for crossover.
     '''

    def __init__(self, input_encoder, population_size, num_hidden_layers=1, satisfied_reward_factor=2):
        strategy = FitnessShapingControl(input_encoder, "", 4, training=True, num_hidden_layers=num_hidden_layers)
        super().__init__(input_encoder, population_size, strategy, num_hidden_layers=num_hidden_layers)

    def create_population(self, problem):
        self.population = Population.random(problem, self.population_size)
        self.evaluation_function(self.population)
        self.problem = problem

    def perform_one_generation(self, generations_left):
        self.population.local_search(5, 1)

        # Modify fitness and perform crossover
        self.evaluation_function(self.population)
        state = self.input_encoder.encode(self.population, generations_left)
        fitness_factors = self.strategy.select_action(state).data.cpu().numpy()[0]
        self.population.modify_fitness(fitness_factors)
        self.population.crossover(n_best=int(self.population_size / 2))
        self.evaluation_function(self.population)

        self.population.selection(self.population_size)
        self.population.mutate(np.repeat(0.05, self.population_size))
        self.evaluate_and_reward(generations_left)

class SolverWithFitnessShapingSelection(SatSolver):
    '''
     Solver that uses the network to control selection.
     The network outputs factors for each individuals, which are multiplied with the individual fitness.
     '''

    def __init__(self, input_encoder, population_size, num_hidden_layers=1, satisfied_reward_factor=2):
        strategy = FitnessShapingControl(input_encoder, "", 4, training=True, num_hidden_layers=num_hidden_layers)
        super().__init__(input_encoder, population_size, strategy, num_hidden_layers=num_hidden_layers)

    def create_population(self, problem):
        self.population = Population.random(problem, self.population_size)
        self.evaluation_function(self.population)
        self.problem = problem

    def perform_one_generation(self, generations_left, factors_log=None):
        self.population.local_search(5, 1)
        self.population.crossover(n_best=int(self.population_size / 2))

        # Modify fitness and perform selection
        self.evaluation_function(self.population)
        state = self.input_encoder.encode(self.population, generations_left)
        fitness_factors = self.strategy.select_action(state).data.cpu().numpy()[0]
        self.population.modify_fitness(fitness_factors)
        self.population.selection(self.population_size)
        self.evaluation_function(self.population)

        # Log mutation rates to file
        if factors_log is not None:
            sorted = []
            for rate in fitness_factors:
                sorted.append(rate)

            sorted.sort()
            for rate in sorted:
                factors_log.write(str(rate) + ' ')

            factors_log.write('\n')
            factors_log.flush()

        self.population.mutate(np.repeat(0.05, self.population_size))
        self.evaluate_and_reward(generations_left)

class VanilaSolver(SatSolver):
    '''
    Solver without a network that performs the standard genetic algorithm and local search with fixed parameters
     '''
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_score = 0

    def create_population(self, problem):
        self.population = Population.random(problem, self.population_size)
        self.evaluation_function(self.population)
        self.problem = problem

    def perform_one_generation(self, _):
        self.population.local_search(5, 1)
        self.population.crossover(n_best=int(self.population_size / 2))
        self.population.selection(self.population_size)
        self.population.mutate(np.repeat(self.mutation_rate, self.population_size))

        self.evaluation_function(self.population)

        score = self.population.get_best_solution().get_score()
        if self.best_score == 0:
            self.best_score = score

        if score > self.best_score:
            self.best_score = score