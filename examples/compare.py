import csv
import numpy as np
import logging
from scipy.stats import qmc
from scipy.spatial.distance import pdist
import sys
import time

from sfsfd import sampling_model

# Activate info-level logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Create a SF-SFD model object
model = sampling_model.SamplingModel( 
    dimension_of_input_space=20,            # This determines the dimension of the sampling problem defined in the WSC paper, but NOT THE OPTIMIZATION PROBLEM
    grid_size=10,  # This determines the dimension of the optimization problem (2*grid_size-1) in our paper we always used size 10, probably can get better results with a bigger number
    no_of_iterations_per_perturbation = 50, # Number of MC samples taken to evaluate objective
    adaptive_sample_size = 10,              # Every this many iterations, increase the sample size by 1 (when 0, it stays constant)
    sample_size = 100,                      # For the original problem, we are tuning a distribution so we can draw 100 points in R^20
    weights = np.eye(3)[0]                  # Do not change this value
)

# Fixes the random seed for reproducability
np.random.seed(0)

# Reset the model and save results to file_name_csv
angle_array = model.initialize()
bounds = [(0.0, 2*np.pi) for i in range(len(angle_array))] # All variables could range from 0 to 2*pi

# Tick
start = time.time()
# Optimize with COBYLA solver
from scipy import optimize
res = optimize.minimize(
    fun = model.iterative_step, 
    x0 = angle_array, 
    method = 'COBYLA',
    #bounds = bounds,  # I guess COBYLA does not accept bounds, so we did not use them
    options={'maxiter': 1000}
)
# Tock
walltime = time.time() - start
# Write row of results to a csv
with open("results.csv", "a") as fp:
    csvwriter = csv.writer(fp)
    csvwriter.writerow(["COBYLA", model.final_exp_disc, walltime])
