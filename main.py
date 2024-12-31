import numpy as np
from scipy import stats
from GeneticAlgorithm import Problem, GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from problems import setup_g1_problem, setup_g4_problem, setup_g5_problem, setup_g6_problem
from tabulate import tabulate
from HelperFunctions import compare_algorithms, run_multiple_trials, printDetailedResults, writeDetailedResultsToFile


if __name__ == "__main__":
    problems = {
        'G1': setup_g1_problem(),
        'G4': setup_g4_problem(),
        'G5': setup_g5_problem(),
        'G6': setup_g6_problem()
    }
    
    results_table = []
    
    for name, problem in problems.items():
        print(f"\n=== Comparing algorithms on {name} problem ===")
        print("\n=== Running Using Genetic Algorithm ===")
        resultsGA, statsGA, objectiveValuesGA = run_multiple_trials(problem, GeneticAlgorithm, num_trials=30)

        print(objectiveValuesGA)

        # Write detailed results to file
        writeDetailedResultsToFile(resultsGA, statsGA, objectiveValuesGA, name, "Genetic Algorithm")

        print("\n=== Running Differential Evolution ===")
        resultsDE, statsDE, objectiveValuesDE = run_multiple_trials(problem, DifferentialEvolution, num_trials=30)
        
        print(objectiveValuesDE)
        # Write detailed results to file
        writeDetailedResultsToFile(resultsDE, statsDE, objectiveValuesDE, name, "Differential Evolution")

        print("\n=== Comparing Using Wilcoxon Rank-Sum Test ===")

        results = compare_algorithms(objectiveValuesGA, objectiveValuesDE)
        
        if results:
            results_table.append([
                name,
                f"{statsGA['mean']:.6f} ± {statsGA['std']:.6f}",
                f"{statsGA['feasible_ratio']*100:.1f}%",
                f"{statsDE['mean']:.6f} ± {statsDE['std']:.6f}",
                f"{statsDE['feasible_ratio']*100:.1f}%",
                f"{results['p_value']:.6f}",
                f"{results['effect_size']:.3f}"
            ])
        else:
            results_table.append([name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    
    # Print comparison table
    headers = ["Problem", "GA (Mean ± Std)", "GA Feasible", 
              "DE (Mean ± Std)", "DE Feasible", "p-value", "Effect Size"]
    print("\nComparison Results:")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))