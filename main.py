"""
Copyright (c) 2025 by IchinoseHimeki(darwinlee1998@gmail.com)
Open source according to the terms of the GPLv3 license.
This script implements a multi-objective optimization problem for agricultural land use
allocation using the NSGA-III algorithm. The problem is defined in three levels, each with its own
objective functions and constraints. The optimization process is carried out using the pymoo library.
The script includes the following steps:
1. Import necessary libraries and modules.
2. Define the number of regions and crops.
3. Generate random data for land and water limits, crop parameters, and environmental constraints.
4. Define the AgriculturalOptimization class, which inherits from the Problem class.
5. Implement the _evaluate method to calculate the objective functions and constraints.
6. Define the compute_objectives function to compute the objectives based on the decision variables.
7. Define the Level1AgriculturalOptimization, Level2AgriculturalOptimization, and Level3AgriculturalOptimization
   classes, each representing a different level of the optimization problem.
8. Implement the _evaluate method for each level to calculate the respective objective functions and constraints.
9. Set up the NSGA-III algorithm for each level and run the optimization process.
10. Print the best objective values for each level.
11. Save the final results to an Excel file.
12. Visualize the distribution of the ecological impact (f3) for the final level.
"""
# -*- coding: utf-8 -*-
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from datareader import DataReader
import pandas as pd

n_regions = 41
n_crops = 3

land_limits = np.random.uniform(100, 500, n_regions)
water_limits = np.random.uniform(1000, 5000, n_regions)

P = np.random.uniform(1000, 5000, (n_regions, n_crops))
W = np.random.uniform(500, 1500, (n_regions, n_crops))
D = np.random.uniform(1, 5, (n_regions, n_crops))
E = np.random.uniform(5000, 15000, n_regions)

n_regions, n_crops, land_limits, water_limits, P, W, D, E = DataReader()

print("P:", P)
print("W:", W)
print("D:", D)
print("E:", E)
print("land_limits:", land_limits)
print("water_limits:", water_limits)

class AgriculturalOptimization(Problem):
    def __init__(self):
        n_vars = n_regions * n_crops
        xl = np.zeros(n_vars)
        xu = np.tile(land_limits, n_crops)
        indices_to_fix = [21 * n_crops + 0, 40 * n_crops + 0]
        for idx in indices_to_fix:
            xl[idx] = 0
            xu[idx] = 0
        super().__init__(n_var=n_vars,
                         n_obj=3,
                         n_constr=2 * n_regions,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        x = x.reshape(n_samples, n_regions, n_crops)
        x[:, 21, 0] = 0
        x[:, 40, 0] = 0
        f1 = -np.sum(P * x, axis=(1, 2))
        f2 = np.sum(W * x, axis=(1, 2))
        f3 = np.sum((E - np.sum(D * x, axis=2)).clip(0), axis=1)
        g1 = np.sum(x, axis=2) - land_limits
        g2 = np.sum(W * x, axis=2) - water_limits
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

def compute_objectives(x):
    n_samples = x.shape[0]
    x_reshaped = x.reshape(n_samples, n_regions, n_crops)
    x_reshaped[:, 21, 0] = 0
    x_reshaped[:, 40, 0] = 0
    f1 = -np.sum(P * x_reshaped, axis=(1, 2))
    f2 = np.sum(W * x_reshaped, axis=(1, 2))
    f3 = np.sum((E - np.sum(D * x_reshaped, axis=2)).clip(0), axis=1)
    return f1, f2, f3

class Level1AgriculturalOptimization(Problem):
    def __init__(self):
        n_vars = n_regions * n_crops
        xl = np.zeros(n_vars)
        xu = np.tile(land_limits, n_crops)
        indices_to_fix = [21 * n_crops + 0, 40 * n_crops + 0]
        for idx in indices_to_fix:
            xl[idx] = 0
            xu[idx] = 0
        super().__init__(n_var=n_vars,
                         n_obj=1,
                         n_constr=2 * n_regions,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        x = x.reshape(n_samples, n_regions, n_crops)
        x[:, 21, 0] = 0
        x[:, 40, 0] = 0
        f1 = -np.sum(P * x, axis=(1, 2))
        g1 = np.sum(x, axis=2) - land_limits
        g2 = np.sum(W * x, axis=2) - water_limits
        out["F"] = np.column_stack([f1])
        out["G"] = np.column_stack([g1, g2])

algorithm_level1 = NSGA3(pop_size=92,
                         ref_dirs=get_reference_directions("das-dennis", 1, n_partitions=12))
level1_problem = Level1AgriculturalOptimization()
res_level1 = minimize(level1_problem,
                      algorithm_level1,
                      ('n_gen', 20000),
                      seed=1,
                      verbose=True)
best_f1 = np.min(res_level1.F)
epsilon = 1e-6
print("Level 1 best f1:", best_f1)

class Level2AgriculturalOptimization(Problem):
    def __init__(self, f1_upper):
        n_vars = n_regions * n_crops
        xl = np.zeros(n_vars)
        xu = np.tile(land_limits, n_crops)
        indices_to_fix = [21 * n_crops + 0, 40 * n_crops + 0]
        for idx in indices_to_fix:
            xl[idx] = 0
            xu[idx] = 0
        super().__init__(n_var=n_vars,
                         n_obj=1,
                         n_constr=2 * n_regions + 1,
                         xl=xl,
                         xu=xu)
        self.f1_upper = f1_upper

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, n_regions, n_crops)
        x_reshaped[:, 21, 0] = 0
        x_reshaped[:, 40, 0] = 0
        f1 = -np.sum(P * x_reshaped, axis=(1, 2))
        f2 = np.sum(W * x_reshaped, axis=(1, 2))
        g1 = np.sum(x_reshaped, axis=2) - land_limits
        g2 = np.sum(W * x_reshaped, axis=2) - water_limits
        g_f1 = f1 - self.f1_upper
        out["F"] = np.column_stack([f2])
        out["G"] = np.column_stack([g1, g2, g_f1])

algorithm_level2 = NSGA3(pop_size=92,
                         ref_dirs=get_reference_directions("das-dennis", 1, n_partitions=12))
level2_problem = Level2AgriculturalOptimization(best_f1 + epsilon)
res_level2 = minimize(level2_problem,
                      algorithm_level2,
                      ('n_gen', 20000),
                      seed=1,
                      verbose=True)
best_f2 = np.min(res_level2.F)
print("Level 2 best f2:", best_f2)

class Level3AgriculturalOptimization(Problem):
    def __init__(self, f1_upper, f2_upper):
        n_vars = n_regions * n_crops
        xl = np.zeros(n_vars)
        xu = np.tile(land_limits, n_crops)
        indices_to_fix = [21 * n_crops + 0, 40 * n_crops + 0]
        for idx in indices_to_fix:
            xl[idx] = 0
            xu[idx] = 0
        super().__init__(n_var=n_vars,
                         n_obj=1,
                         n_constr=2 * n_regions + 2,
                         xl=xl,
                         xu=xu)
        self.f1_upper = f1_upper
        self.f2_upper = f2_upper

    def _evaluate(self, x, out, *args, **kwargs):
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, n_regions, n_crops)
        x_reshaped[:, 21, 0] = 0
        x_reshaped[:, 40, 0] = 0
        f1 = -np.sum(P * x_reshaped, axis=(1, 2))
        f2 = np.sum(W * x_reshaped, axis=(1, 2))
        f3 = np.sum((E - np.sum(D * x_reshaped, axis=2)).clip(0), axis=1)
        g1 = np.sum(x_reshaped, axis=2) - land_limits
        g2 = np.sum(W * x_reshaped, axis=2) - water_limits
        g_f1 = f1 - self.f1_upper
        g_f2 = f2 - self.f2_upper
        out["F"] = np.column_stack([f3])
        out["G"] = np.column_stack([g1, g2, g_f1, g_f2])

algorithm_level3 = NSGA3(pop_size=92,
                         ref_dirs=get_reference_directions("das-dennis", 1, n_partitions=12))
level3_problem = Level3AgriculturalOptimization(best_f1 + epsilon, best_f2 + epsilon)
res_level3 = minimize(level3_problem,
                      algorithm_level3,
                      ('n_gen', 20000),
                      seed=1,
                      verbose=True)
best_f3 = np.min(res_level3.F)
print("Level 3 best f3:", best_f3)
print("\n=== Final Solution ===")
print("Level 1 best f1:", best_f1)
print("Level 2 best f2:", best_f2)
print("Level 3 best f3:", best_f3)

if res_level3.X.ndim == 1:
    res_X_reshaped = res_level3.X.reshape(1, n_regions, n_crops)
else:
    res_X_reshaped = res_level3.X.reshape(res_level3.X.shape[0], n_regions, n_crops)

df_F = pd.DataFrame(res_level3.F, columns=["Ecological Impact"])
with pd.ExcelWriter("final_optimization_results.xlsx") as writer:
    df_F.to_excel(writer, sheet_name="Final Objectives", index=False)
    for i in range(res_X_reshaped.shape[0]):
        df_X = pd.DataFrame(res_X_reshaped[i], columns=[f"Crop_{j+1}" for j in range(n_crops)])
        df_X.insert(0, "Region", [f"Region_{k+1}" for k in range(n_regions)])
        df_X.to_excel(writer, sheet_name=f"Decision Variables_{i+1}", index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(range(len(res_level3.F)), res_level3.F, c='blue')
plt.xlabel("Individual Index")
plt.ylabel("Ecological Impact (f3)")
plt.title("Level 3 Ecological Impact Distribution")
plt.show()