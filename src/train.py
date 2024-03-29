from utils.solve import start

import torch
import random
import sys


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# random.seed(0)
print(sys.argv)

##############################################################
##                  Possible Arguments                      ##
##############################################################

solver_arg = sys.argv[1]
outdir = sys.argv[2]

# directory for baseline file
weightsdir = None
# e.g. pre:INDEX to start with a specific task
start_at = None
if len(sys.argv) > 3:
    weightsdir = sys.argv[3]
if len(sys.argv) > 4:
    start_at = sys.argv[4]

start(True, solver_arg, outdir, weightsdir, start_at)
