# Libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

def initializePopulation(miu, numVariables, bounds):
    """
    Initialize population with different bounds for each variable
    """
    population = np.zeros((miu, numVariables))
    for i in range(numVariables):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], miu)
    return population

def G4(x):
    """
    G4 benchmark function
    x: A 5-d vector
    """
    return 5.3578547 * x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141

def checkBoundsViolation(x, bounds):
    """
    Check if solution violates any variable bounds
    Returns True if bounds are violated, False otherwise
    """
    for i, (lower, upper) in enumerate(bounds):
        if x[i] < lower or x[i] > upper:
            return True
    return False

def evaluateConstraints(x):
    """
    Evaluate all constraints for G4 and return their violations
    For double-sided constraints, we return two values (one for each side)
    Returns positive values for violations, 0 otherwise
    """
    # Constraint 1: 0 <= g1 <= 92
    g1 = 85.334407 + 0.0056858*x[1]*x[4] + 0.00026*x[0]*x[3] - 0.0022053*x[2]*x[4]
    c1_lower = max(0, -g1)  # Violation if g1 < 0
    c1_upper = max(0, g1 - 92)  # Violation if g1 > 92
    
    # Constraint 2: 90 <= g2 <= 110
    g2 = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813 * x[2]**2
    c2_lower = max(0, 90 - g2)  # Violation if g2 < 90
    c2_upper = max(0, g2 - 110)  # Violation if g2 > 110
    
    # Constraint 3: 20 <= g3 <= 25
    g3 = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]
    c3_lower = max(0, 20 - g3)  # Violation if g3 < 20
    c3_upper = max(0, g3 - 25)  # Violation if g3 > 25
    
    return {
        'g1_lower': c1_lower, 'g1_upper': c1_upper,
        'g2_lower': c2_lower, 'g2_upper': c2_upper,
        'g3_lower': c3_lower, 'g3_upper': c3_upper
    }

def dynamicPenaltyFunction(violations, generation, max_generations):
    """
    Calculate penalty based on:
    1. Current generation (increases over time)
    2. Magnitude of violation
    3. Number of constraints violated
    """
    # Convert dictionary values to array for processing
    violation_values = np.array(list(violations.values()))
    
    if np.sum(violation_values) == 0:
        return 0
    
    # Dynamic penalty coefficient increases with generations
    C = 2 * (generation / max_generations)
    
    # Penalty grows quadratically with violation magnitude
    violation_penalty = np.sum(violation_values ** 2)
    
    # Additional penalty for number of violated constraints
    num_violations = np.sum(violation_values > 0)
    violation_count_penalty = num_violations * 10
    
    return (C + 1) * (violation_penalty + violation_count_penalty)

def getConstraintValues(x):
    """
    Get actual constraint values for reporting
    """
    g1 = 85.334407 + 0.0056858*x[1]*x[4] + 0.00026*x[0]*x[3] - 0.0022053*x[2]*x[4]
    g2 = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813 * x[2]**2
    g3 = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]
    
    return {'g1': g1, 'g2': g2, 'g3': g3}

def analyzeViolations(x, bounds):
    """
    Analyze and report all violations in detail
    """
    violations = []
    
    # Check bounds violations
    for i, (lower, upper) in enumerate(bounds):
        if x[i] < lower:
            violations.append(f"x[{i}] = {x[i]:.6f} violates lower bound {lower}")
        elif x[i] > upper:
            violations.append(f"x[{i}] = {x[i]:.6f} violates upper bound {upper}")
    
    # Check constraint violations
    constraint_values = getConstraintValues(x)
    if constraint_values['g1'] < 0 or constraint_values['g1'] > 92:
        violations.append(f"g1 = {constraint_values['g1']:.6f} outside [0, 92]")
    if constraint_values['g2'] < 90 or constraint_values['g2'] > 110:
        violations.append(f"g2 = {constraint_values['g2']:.6f} outside [90, 110]")
    if constraint_values['g3'] < 20 or constraint_values['g3'] > 25:
        violations.append(f"g3 = {constraint_values['g3']:.6f} outside [20, 25]")
    
    return violations

def obtainFitness(population, function, generation, max_generations):
    """
    Modified fitness function incorporating constraints
    """
    fitness = []
    for ind in population:
        # Calculate objective function value
        obj_value = function(ind)
        
        # Calculate constraint violations
        violations = evaluateConstraints(ind)
        
        # Calculate penalty
        penalty = dynamicPenaltyFunction(violations, generation, max_generations)
        
        # Final fitness is objective value minus penalty
        # We add a small constant to avoid division by zero in selection
        fitness.append(1.0 / (obj_value + penalty + 1e-10))
    
    return np.array(fitness)

def parameterBasedMutation(individual, pm, bounds, nm=20):
    """
    Modified mutation operator that respects variable-specific bounds
    """
    if np.random.rand() <= pm:
        position = np.random.randint(0, len(individual))
        lowerBound, upperBound = bounds[position]
        
        u = np.random.uniform(0, 1)
        delta = np.min([(individual[position] - lowerBound), 
                       (upperBound - individual[position])]) / (upperBound - lowerBound)
        eta_m = nm
        
        if u <= 0.5:
            delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
        else:
            delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1))
            
        individual[position] = individual[position] + delta_q * (upperBound - lowerBound)
        individual[position] = np.clip(individual[position], lowerBound, upperBound)
    
    return individual
    

def geneticAlgorithmG4(miu=200, numGenerations=500, pc=0.9, pm=0.1, verbose=False):
    """
    Genetic Algorithm specifically configured for G4 problem
    """
    # Define bounds for each variable
    bounds = [
        (78, 102),    # x0
        (33, 45),     # x1
        (27, 45),     # x2
        (27, 45),     # x3
        (27, 45)      # x4
    ]
    
    numVariables = 5
    
    # Initialize population
    population = initializePopulation(miu, numVariables, bounds)
    bestPerGeneration = []
    bestIndividualperGeneration = []
    feasibleFound = False
    bestFeasibleSolution = None
    bestFeasibleFitness = float('inf')  # Changed to inf because G4 is a minimization problem

    for generation in range(numGenerations):
        # Obtain fitness for the current population
        fitness = obtainFitness(population, G4, generation, numGenerations)
        
        # Track best solution
        best_idx = np.argmax(fitness)
        current_best = population[best_idx]
        
        # Check if current best is feasible
        violations = evaluateConstraints(current_best)
        if all(v <= 1e-6 for v in violations.values()):
            feasibleFound = True
            current_value = G4(current_best)
            if bestFeasibleSolution is None or current_value < bestFeasibleFitness:
                bestFeasibleSolution = current_best.copy()
                bestFeasibleFitness = current_value
        
        bestPerGeneration.append(np.max(fitness))
        bestIndividualperGeneration.append(current_best)
        
        # Create new population
        newPopulation = []
        for _ in range(miu // 2):
            parent1 = binaryTournamentSelection(population, fitness)
            parent2 = binaryTournamentSelection(population, fitness)
            
            child1, child2 = simulatedBinaryCrossover(parent1, parent2, pc)
            
            # Apply bounds-aware mutation
            child1 = parameterBasedMutation(child1, pm, bounds)
            child2 = parameterBasedMutation(child2, pm, bounds)
            
            newPopulation.extend([child1, child2])
        
        population = np.array(newPopulation)
    
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(range(numGenerations), bestPerGeneration)
        plt.title("Convergence Graph")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.show()
    
    if feasibleFound:
        return bestFeasibleSolution, bestFeasibleFitness, bestPerGeneration
    else:
        bestIndividual = bestIndividualperGeneration[np.argmax(bestPerGeneration)]
        return bestIndividual, G4(bestIndividual), bestPerGeneration

def runMultipleGA(num_runs=30, miu=200, numGenerations=500, pc=0.9, pm=0.1):
    """
    Run multiple independent executions of the GA and analyze results
    """
    bounds = [
        (78, 102),    # x0
        (33, 45),     # x1
        (27, 45),     # x2
        (27, 45),     # x3
        (27, 45)      # x4
    ]
    
    results = []
    objective_values = []
    feasible_solutions = 0
    
    for run in range(num_runs):
        print(f"Running execution {run + 1}/{num_runs}")
        best_solution, best_value, _ = geneticAlgorithmG4(
            miu=miu, 
            numGenerations=numGenerations, 
            pc=pc, 
            pm=pm, 
            verbose=False
        )
        
        # Check feasibility
        violations = evaluateConstraints(best_solution)
        is_feasible = all(v <= 1e-6 for v in violations.values()) and not checkBoundsViolation(best_solution, bounds)
        
        if is_feasible:
            feasible_solutions += 1
            
        results.append({
            'run': run + 1,
            'solution': best_solution,
            'objective_value': best_value,
            'feasible': is_feasible,
            'violations': analyzeViolations(best_solution, bounds)
        })
        
        objective_values.append(best_value)
    
    # Statistical analysis
    objective_values = np.array(objective_values)
    stats_results = {
        'mean': np.mean(objective_values),
        'std': np.std(objective_values),
        'min': np.min(objective_values),
        'max': np.max(objective_values),
        'feasible_ratio': feasible_solutions / num_runs
    }
    
    return results, stats_results

def binaryTournamentSelection(population, fitness):
    """
    Binary tournament selection (deterministic)
    """
    i, j = np.random.choice(len(population), 2, replace=False)
    if fitness[i] > fitness[j]:
        return population[i].copy()
    else:
        return population[j].copy()

def simulatedBinaryCrossover(parent1, parent2, pc, eta_c=20):
    """
    Simulated Binary Crossover (SBX)
    """
    if np.random.rand() <= pc:
        child1, child2 = np.copy(parent1), np.copy(parent2)
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:
                beta = (2 * np.random.rand()) ** (1 / (eta_c + 1))
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        return child1, child2
    return parent1.copy(), parent2.copy()

def printDetailedResults(results, stats_results):
    """
    Print detailed results in a formatted way
    """
    print("\n=== G4 Problem Multiple Runs Analysis ===\n")
    
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
            r['run'],
            f"{r['objective_value']:.6f}",
            "Yes" if r['feasible'] else "No",
            len(r['violations']),
            r["solution"]
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print violation details for non-feasible solutions
    print("\nViolation Details:")
    for r in results:
        if not r['feasible']:
            print(f"\nRun {r['run']}:")
            for v in r['violations']:
                print(f"  - {v}")

# Example usage
if __name__ == "__main__":
    # Run multiple GA executions
    results, stats = runMultipleGA(num_runs=30)
    
    # Print detailed results
    printDetailedResults(results, stats)
    
    # Plot histogram of objective values
    plt.figure(figsize=(10, 6))
    plt.hist([r['objective_value'] for r in results], bins=15)
    plt.title("Distribution of Objective Values")
    plt.xlabel("Objective Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
