# Genetic Algorithm for the Boolean Satisfiability Problem

This project is an attempt to use a neural network to improve genetic algorithms for the boolean satisfiability problem.
The code is mostly based on the source code of "Learning to Evolve, Jan Schuchardt, Vladimir Golkov, Daniel Cremers [2019]".

## Code
### neural_networks
Code for instantiating and working with PyTorch networks. Taken from Jan Schuchardt's code base and left unchanged.

### presentation
Test results included in the final presentation, training and validation logs

### reinforcement
Code for reinforcement learning. Taken from Jan Schuchardt's code based and left unchanged, except allowing for 
irregular sized training data batches.

### sat
Own code for representing and validating SAT problems
* cnf3.py -
Code for reading CNF3 problems and evaluating them.
* population.py -
Code for representing and modifying a population of CNF3 solutions.
* problem loader.py -
Loads SAT3 problems from text files.

#### Solution
Representing a single CNF3 solution.

### Solvers
Code for encoding populations and problems for neural networks and running the genetic algorithm.
Implementation of new classes based on Jan Schuchardt's code base.

### Strategies
Code for neural network based startegies, used as part of a solver to replace one step. Implementation of new classes based on Jan Schuchardt's code base.

### utils
Utility code for training and validation.

### examples-easy, examples-50, examples-75
Test instances from https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html

## How to Use
Four different solvers are available:
* Individual Mutation
* Gene Mutation
* Crossover
* Selection

They can be trained using _train.py_ with the following syntax

`python3 train.py [gene/individual/crossover/selection] [output directory] [weights directory]? [start index]? `

They can be validated using _validation_py with the following syntax (network weights are expected in the output dir as "baseline")

`python3 validate.py [gene/individual/crossover/selection] [output directory]`

The validation outputs the average maximum fitness at each generation to a text files at the output direction.

# Slides
See _presentation.pdf_

# Future Work
See _future work_ section of the slides.