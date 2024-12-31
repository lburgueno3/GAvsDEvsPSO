import numpy as np
from tabulate import tabulate
from scipy import stats
from GeneticAlgorithm import Problem, GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
import matplotlib.pyplot as plt

def run_multiple_trials(problem: Problem, algorithmToUse, num_trials: int = 30, plot_convergence: bool = False,  **params):
    """Run multiple trials and analyze results"""
    results = []
    objective_values = []
    plot_data = []

    if algorithmToUse == GeneticAlgorithm:
        algorithm_name = "GA"
    else:
        algorithm_name = "DE"
    
    if plot_convergence:
        cols = min(6, num_trials)
        rows = (num_trials + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        fig.suptitle(f'Convergence Plot for {num_trials} Trials, Using {algorithm_name}')
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        algorithm = algorithmToUse(problem, plot_convergence=plot_convergence, **params)
        best_solution, obj_value, current_plot_data = algorithm.run()
        
        # Check feasibility
        total_violation, violations = algorithm.evaluate_constraints(best_solution)
        is_feasible = total_violation <= 1e-6
        
        results.append({
            'trial': trial + 1,
            'solution': best_solution,
            'objective_value': obj_value,
            'feasible': is_feasible,
            'violation': total_violation,
            'violations': violations
        })
        
        objective_values.append(obj_value)
        plot_data.append(current_plot_data)

        # Make the plot for this trial
        if plot_convergence:
            row = trial // cols
            col = trial % cols
            ax = axes[row, col]
            ax.plot(current_plot_data, 'b-', linewidth=1)
            ax.set_title(f'Trial {trial + 1}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Objective Value')
            ax.grid(True)

    if plot_convergence:
        # Remove any additional empty subplots
        for i in range(trial + 1, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()

    
    # Calculate statistics
    stats = {
        'mean': np.mean(objective_values),
        'std': np.std(objective_values),
        'min': np.min(objective_values),
        'max': np.max(objective_values),
        'feasible_ratio': sum(1 for r in results if r['feasible']) / num_trials
    }
    
    return results, stats, objective_values

def writeDetailedResultsToFile(results, stats_results, objective_values, problemName, algorithmUsed, filename="detailed_results.txt"):
    """
    Write detailed results to a file in a formatted way.
    """
    with open(filename, "a") as file:
        file.write(f"=== Problem {problemName} using {algorithmUsed} ===")
        file.write("\n=== Multiple Runs Analysis ===\n\n")
        
        # Write summary statistics
        file.write("Summary Statistics:\n")
        file.write(f"Mean Objective Value: {stats_results['mean']:.6f}\n")
        file.write(f"Std Dev Objective Value: {stats_results['std']:.6f}\n")
        file.write(f"Best Objective Value: {stats_results['min']:.6f}\n")
        file.write(f"Worst Objective Value: {stats_results['max']:.6f}\n")
        file.write(f"Feasible Solutions: {stats_results['feasible_ratio']*100:.1f}%\n\n")
        
        # Write detailed results for each run
        file.write("Detailed Results:\n")
        headers = ["Run", "Objective Value", "Feasible", "Degree of Violation", "# of Violations", "Best Solution Values"]
        table_data = []
        
        for r in results:
            table_data.append([
                r['trial'],
                f"{r['objective_value']:.6f}",
                "Yes" if r['feasible'] else "No",
                r['violation'],
                sum(x != 0 for x in r['violations']), #non-zero elements in the list of violations
                r["solution"]
            ])
        
        file.write(tabulate(table_data, headers=headers, tablefmt="grid") + "\n")

        file.write(f"Objective Values: {objective_values}")
        
        # Write violation details for non-feasible solutions
        file.write("\nViolation Details:\n")
        for r in results:
            if not r['feasible']:
                file.write(f"\nRun {r['trial']}:\n")
                for v in r['violations']:
                    file.write(f"  - {v}\n")



def printDetailedResults(results, stats_results):
    """
    Print detailed results in a formatted way
    """
    print("\n=== Multiple Runs Analysis ===\n")
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"Mean Objective Value: {stats_results['mean']:.6f}")
    print(f"Std Dev Objective Value: {stats_results['std']:.6f}")
    print(f"Best Objective Value: {stats_results['min']:.6f}")
    print(f"Worst Objective Value: {stats_results['max']:.6f}")
    print(f"Feasible Solutions: {stats_results['feasible_ratio']*100:.1f}%")
    
    # Print detailed results for each run
    print("\nDetailed Results:")
    headers = ["Run", "Objective Value", "Feasible", "# Violations", "Best Solution Values"]
    table_data = []
    
    for r in results:
        table_data.append([
            r['trial'],
            f"{r['objective_value']:.6f}",
            "Yes" if r['feasible'] else "No",
            r['violation'],
            r["solution"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print violation details for non-feasible solutions
    print("\nViolation Details:")
    for r in results:
        if not r['feasible']:
            print(f"\nRun {r['trial']}:")
            for v in r['violations']:
                print(f"  - {v}")

def compare_algorithms(ga_results, de_results):
    """Compare GA and DE performance using Wilcoxon rank-sum test"""   
    # Perform statistical test
    statistic, p_value = stats.ranksums(ga_results, de_results)
    
    # Calculate effect size (r = Z / sqrt(N))
    n1, n2 = len(ga_results), len(de_results)
    effect_size = abs(statistic) / np.sqrt(n1 + n2)
    
    return {
        'p_value': p_value,
        'effect_size': effect_size
    }