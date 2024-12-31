import numpy as np
from typing import List, Callable, Dict, Tuple
import matplotlib.pyplot as plt
from tabulate import tabulate
from GeneticAlgorithm import Problem, GeneticAlgorithm
from constraints import ConstraintType, create_constraint_checker
from problems import setup_g1_problem, setup_g4_problem, setup_g5_problem, setup_g6_problem, setup_layeb05_problem, setup_layeb10_problem, setup_layeb15_problem, setup_layeb18_problem
from HelperFunctions import run_multiple_trials, printDetailedResults, writeDetailedResultsToFile
from DifferentialEvolution import DifferentialEvolution

if __name__ == "__main__":
    # DE
    print("\n=== Results for Layeb Problem ===")
    problem = setup_layeb15_problem()
    resultsDE, statsDE, objectiveValuesDE = run_multiple_trials(problem, DifferentialEvolution, num_trials=3, plot_convergence=True)
    
    print(f"Mean objective value: {statsDE['mean']:.6f}")
    print(f"Std Dev: {statsDE['std']:.6f}")
    print(f"Best solution found: {statsDE['min']:.6f}")
    print(f"Feasible solutions: {statsDE['feasible_ratio']*100:.1f}%")

    # Print detailed results
    printDetailedResults(resultsDE, statsDE)
    writeDetailedResultsToFile(resultsDE, statsDE, objectiveValuesDE, "Layeb", "Differential Evolution")
    
    # GA
    print("\n=== Results for layeb Problem ===")
    problem = setup_layeb15_problem()
    resultsGA, statsGA, objectiveValuesGA = run_multiple_trials(problem, GeneticAlgorithm, num_trials=3, plot_convergence=True)
    
    print(f"Mean objective value: {statsGA['mean']:.6f}")
    print(f"Std Dev: {statsGA['std']:.6f}")
    print(f"Best solution found: {statsGA['min']:.6f}")
    print(f"Feasible solutions: {statsGA['feasible_ratio']*100:.1f}%")

    # Print detailed results
    printDetailedResults(resultsGA, statsGA)
    writeDetailedResultsToFile(resultsGA, statsGA, objectiveValuesGA, "Layeb", "Genetic Algorithm")

    

    """# G1
    print("\n=== Results for G1 Problem ===")
    g1_problem = setup_g1_problem()
    resultsG1GA, statsG1GA, objectiveValuesGA = run_multiple_trials(g1_problem, GeneticAlgorithm, num_trials=3)
    
    print(f"Mean objective value: {statsG1GA['mean']:.6f}")
    print(f"Std Dev: {statsG1GA['std']:.6f}")
    print(f"Best solution found: {statsG1GA['min']:.6f}")
    print(f"Feasible solutions: {statsG1GA['feasible_ratio']*100:.1f}%")

    # Print detailed results
    printDetailedResults(resultsG1GA, statsG1GA)
    writeDetailedResultsToFile(resultsG1GA, statsG1GA, objectiveValuesGA)"""

    """# G4
    print("\n=== Results for G4 Problem ===")
    g4_problem = setup_g4_problem()
    resultsG4GA, statsG4GA = run_multiple_trials(g4_problem, num_trials=30)
    
    print(f"Mean objective value: {statsG4GA['mean']:.6f}")
    print(f"Std Dev: {statsG4GA['std']:.6f}")
    print(f"Best solution found: {statsG4GA['min']:.6f}")
    print(f"Feasible solutions: {statsG4GA['feasible_ratio']*100:.1f}%")

    # Print detailed results
    printDetailedResults(resultsG4GA, statsG4GA)

    # G5
    print("\n=== Results for G5 Problem ===")
    g5_problem = setup_g5_problem()
    resultsG5GA, statsG5GA = run_multiple_trials(g5_problem, num_trials=30)
    
    print(f"Mean objective value: {statsG5GA['mean']:.6f}")
    print(f"Std Dev: {statsG5GA['std']:.6f}")
    print(f"Best solution found: {statsG5GA['min']:.6f}")
    print(f"Feasible solutions: {statsG5GA['feasible_ratio']*100:.1f}%")

    # Print detailed results
    printDetailedResults(resultsG5GA, statsG5GA)
"""
   