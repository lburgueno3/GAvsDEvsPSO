import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple
from constraints import create_constraint_checker, ConstraintType
from GeneticAlgorithm import Problem

# G1
def setup_g1_problem():
    def g1_objective(x):
        return 5*x[0] + 5*x[1] + 5*x[2] + 5*x[3] - 5 * sum(x[i]**2 for i in range(4)) - sum(x[i] for i in range(4,12))
    
    # Transform constraints to >= 0
    def g1_constraint1(x):
        return 10 - (2*x[0] + 2*x[1] + x[9] + x[10])
    
    def g1_constraint2(x):
        return 10 - (2*x[0] + 2* x[2] + x[9] + x[11])
    
    def g1_constraint3(x):
        return 10 - (2*x[1] + 2*x[2] + x[10] + x[11])
    
    def g1_constraint4(x):
        return -(-8*x[0] + x[9])
    
    def g1_constraint5(x):
        return -(-8 * x[1] + x[10])
    
    def g1_constraint6(x):
        return -(-8* x[2] + x[11])
    
    def g1_constraint7(x):
        return -(-2 * x[3] - x[4] + x[9])
    
    def g1_constraint8(x):
        return -(-2 * x[5] - x[6] + x[10])
    
    def g1_constraint9(x):
        return -(-2 * x[7] - x[8] + x[11])
    
    bounds = [
        (0, 1),    # x0
        (0, 1),    # x1
        (0, 1),    # x2
        (0, 1),    # x3
        (0, 1),    # x4
        (0, 1),    # x5
        (0, 1),    # x6
        (0, 1),    # x7
        (0, 1),    # x8
        (0, 100),    # x9
        (0, 100),    # x10
        (0, 100),    # x11
        (0, 1),    # x12
    ]

    constraints = [
        create_constraint_checker(g1_constraint1, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint2, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint3, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint4, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint5, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint6, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint7, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint8, ConstraintType.INEQUALITY),
        create_constraint_checker(g1_constraint9, ConstraintType.INEQUALITY)
    ]

    return Problem(
        objective_function=g1_objective,
        constraints=constraints,
        bounds=bounds,
        name="G1",
        minimize=True
    )

# G4
def setup_g4_problem():
    def g4_objective(x):
        return 5.3578547 * x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141
    
    def g4_constraint1(x):
        val = 85.334407 + 0.0056858*x[1]*x[4] + 0.00026*x[0]*x[3] - 0.0022053*x[2]*x[4]
        return (0, 92, val)  # left constraint, right constraint, value
    
    def g4_constraint2(x):
        val = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813 * x[2]**2
        return (90, 110, val)
    
    def g4_constraint3(x):
        val = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]
        return (20, 25, val)
    
    bounds = [
        (78, 102),    # x0
        (33, 45),     # x1
        (27, 45),     # x2
        (27, 45),     # x3
        (27, 45)      # x4
    ]
    
    constraints = [
        create_constraint_checker(g4_constraint1, ConstraintType.DOUBLE_SIDED),
        create_constraint_checker(g4_constraint2, ConstraintType.DOUBLE_SIDED),
        create_constraint_checker(g4_constraint3, ConstraintType.DOUBLE_SIDED)
    ]
    
    return Problem(
        objective_function=g4_objective,
        constraints=constraints,
        bounds=bounds,
        name="G4",
        minimize=True
    )

# G5

def setup_g5_problem():
    def g5_objective(x):
        return 3*x[0] + 0.000001*x[0]**3 + 2*x[1] + 0.000002/3 * x[1]**3

    def g5_constraint1(x):
        return x[3] - x[2] + 0.55  

    def g5_constraint2(x):
        return x[2] - x[3] + 0.55  

    # For equality constraints, we'll use tolerance-based checking
    def g5_constraint3(x):
        return 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]

    def g5_constraint4(x):
        return 1000 * np.sin(x[2]-0.25) + 1000 * np.sin(x[2]-x[3]-0.25) + 894.8 - x[1]

    def g5_constraint5(x):
        return 1000 * np.sin(x[3]-0.25) + 1000 * np.sin(x[3]-x[2]-0.25) + 1294.8

    bounds = [
        (0, 1200),    # x0
        (0, 1200),    # x1
        (-0.55, 0.55),  # x2
        (-0.55, 0.55)   # x3
    ]

    constraints = [
        create_constraint_checker(g5_constraint1, ConstraintType.INEQUALITY),
        create_constraint_checker(g5_constraint2, ConstraintType.INEQUALITY),
        create_constraint_checker(g5_constraint3, ConstraintType.EQUALITY),
        create_constraint_checker(g5_constraint4, ConstraintType.EQUALITY),
        create_constraint_checker(g5_constraint5, ConstraintType.EQUALITY)
    ]

    return Problem(
        objective_function=g5_objective,
        constraints=constraints,
        bounds=bounds,
        name="G5",
        minimize=True
    )

# G6

def setup_g6_problem():
    def g6_objective(x):
        return (x[0]-10)**3 + (x[1]-20)**3
    
    def g6_constraint1(x):
        return (x[0]-5)**2 + (x[1]-5)**2 - 100
    
    def g6_constraint2(x):
        return -(x[0]-6)**2 - (x[1]-5)**2 + 82.81
    
    bounds = [
        (13, 100),    # x0
        (0, 100)     # x1
    ]

    constraints = [
        create_constraint_checker(g6_constraint1, ConstraintType.INEQUALITY),
        create_constraint_checker(g6_constraint2, ConstraintType.INEQUALITY)
    ]

    return Problem(
        objective_function=g6_objective,
        constraints=constraints,
        bounds=bounds,
        name="G6",
        minimize=True
    )

# Layeb05
def setup_layeb05_problem():
    def layeb05_objective(x):
        n = len(x)
        result = 0.0
        for i in range(2 - 1):
            numerator = np.log(
                np.abs(np.sin(x[i] - np.pi / 2) + np.cos(x[i + 1] - np.pi)) + 0.001
            )
            denominator = np.abs(np.cos(2 * x[i] - x[i + 1] + np.pi / 2)) + 1
            result += numerator / denominator
        return result
    
    bounds = [
        (-10, 10),    # x0
        (-10, 10)     # x1
    ]

    constraints = []

    return Problem(
        objective_function=layeb05_objective,
        constraints=constraints,
        bounds=bounds,
        name="Layeb05",
        minimize=True
    )

# Layeb10
def setup_layeb10_problem():
    def layeb10_objective(x):
        n = len(x)
        result = 0
        for i in range(2 - 1):
            term1 = np.power(np.log(x[i] * x[i] + x[i + 1] * x[i + 1] + 0.5), 2)
            term2 = np.abs(100 * np.sin(x[i] - x[i + 1]))
            result += term1 + term2
        return result
    
    bounds = [
        (-100, 100),    # x0
        (-100, 100)     # x1
    ]

    constraints = []

    return Problem(
        objective_function=layeb10_objective,
        constraints=constraints,
        bounds=bounds,
        name="Layeb10",
        minimize=True
    )

# Layeb15
def setup_layeb15_problem():
    def layeb15_objective(x):
        n = len(x)
        result = 0
        for i in range(2 - 1):
            arg1 = 2 * np.abs(x[i]) - x[i + 1] * x[i + 1] - 1
            term1 = 10 * np.sqrt(max(0, np.tanh(arg1)))  # Ensure argument for square root is not negative
            term2 = np.clip(np.abs(np.exp(x[i] * x[i + 1] + 1) - 1), -100, 100)
            

            result += term1 + term2
        return result
    
    bounds = [
        (-100, 100),    # x0
        (-100, 100)     # x1
    ]

    constraints = []

    return Problem(
        objective_function=layeb15_objective,
        constraints=constraints,
        bounds=bounds,
        name="Layeb15",
        minimize=True
    )

# Layeb18
def setup_layeb18_problem():
    def layeb18_objective(x):
        result = 0
        for i in range(2 - 1):
            numerator = np.log(np.abs(np.cos(2 * x[i] * x[i + 1] / np.pi)) + 0.001)
            denominator = np.abs(np.sin(x[i] + x[i + 1]) * np.cos(x[i])) + 1
            result += numerator / denominator
        return result
    
    bounds = [
        (-10, 10),    # x0
        (-10, 10)     # x1
    ]

    constraints = []

    return Problem(
        objective_function=layeb18_objective,
        constraints=constraints,
        bounds=bounds,
        name="Layeb18",
        minimize=True
    )