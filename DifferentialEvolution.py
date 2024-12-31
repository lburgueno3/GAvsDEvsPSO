import numpy as np
from GeneticAlgorithm import Problem

class DifferentialEvolution:
    def __init__(self, problem: Problem,
                 population_size: int = 50,
                 num_generations: int = 200,
                 F: float = 0.8,
                 CR: float = 0.9,
                 Pf: float = 0.45,
                 plot_convergence: bool = False):  # Probability factor for stochastic ranking
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.F = F
        self.CR = CR
        self.Pf = Pf
        self.num_variables = len(problem.bounds)
        self.plot_convergence = plot_convergence
    
    def initialize_population(self):
        """Initialize population within bounds"""
        population = np.zeros((self.population_size, self.num_variables))
        for i in range(self.num_variables):
            lower, upper = self.problem.bounds[i]
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population
    
    def evaluate_constraints(self, x):
        """Evaluate all constraints and return total violation"""
        total_violation = 0
        violations = []
        
        for constraint in self.problem.constraints:
            violation = constraint(x)
            violations.append(violation)
            total_violation += violation
            
        return total_violation, violations
    
    def stochastic_ranking(self, population, obj_values, constraint_violations):
        """
        Implement stochastic ranking as shown in the paper's pseudocode
        Returns indices of sorted population
        """
        n = len(population)
        indices = list(range(n))
        
        # Bubble sort with stochastic ranking
        for i in range(n):
            swapped = False
            for j in range(n - 1):
                u = np.random.uniform(0, 1)
                
                # If both solutions are feasible or u < Pf
                if (constraint_violations[indices[j]] == 0 and 
                    constraint_violations[indices[j + 1]] == 0) or u < self.Pf:
                    # Compare by objective function
                    if ((self.problem.minimize and 
                         obj_values[indices[j]] > obj_values[indices[j + 1]]) or
                        (not self.problem.minimize and 
                         obj_values[indices[j]] < obj_values[indices[j + 1]])):
                        # Swap
                        indices[j], indices[j + 1] = indices[j + 1], indices[j]
                        swapped = True
                else:
                    # Compare by constraint violation
                    if constraint_violations[indices[j]] > constraint_violations[indices[j + 1]]:
                        # Swap
                        indices[j], indices[j + 1] = indices[j + 1], indices[j]
                        swapped = True
            
            if not swapped:
                break
                
        return indices
    
    def mutation_best_1(self, population, best_idx):
        """DE/best/1 mutation strategy"""
        mutants = np.zeros_like(population)
        
        for i in range(self.population_size):
            candidates = list(set(range(self.population_size)) - {i, best_idx})
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            mutant = population[best_idx] + self.F * (population[r1] - population[r2])
            
            for j in range(self.num_variables):
                lower, upper = self.problem.bounds[j]
                mutant[j] = np.clip(mutant[j], lower, upper)
            
            mutants[i] = mutant
            
        return mutants
    
    def crossover_bin(self, population, mutants):
        """Binomial crossover"""
        trials = np.zeros_like(population)
        
        for i in range(self.population_size):
            j_rand = np.random.randint(self.num_variables)
            
            for j in range(self.num_variables):
                if np.random.rand() <= self.CR or j == j_rand:
                    trials[i, j] = mutants[i, j]
                else:
                    trials[i, j] = population[i, j]
        
        return trials
    
    def run(self):
        """Run the DE algorithm with stochastic ranking"""
        population = self.initialize_population()
        best_solution = population[0].copy()  # Initialize with first solution
        best_fitness = self.problem.objective_function(best_solution)
        best_violation = self.evaluate_constraints(best_solution)[0]
        best_per_gen = []
        
        for generation in range(self.num_generations):
            # Evaluate population
            obj_values = np.array([self.problem.objective_function(ind) for ind in population])
            constraint_violations = np.array([self.evaluate_constraints(ind)[0] 
                                           for ind in population])
            
            # Sort population using stochastic ranking
            sorted_indices = self.stochastic_ranking(population, obj_values, 
                                                   constraint_violations)
            
            # Update best solution
            for idx in sorted_indices:
                current_violation = constraint_violations[idx]
                current_fitness = obj_values[idx]
                
                # Update if current solution is better
                if (current_violation < best_violation or 
                    (current_violation == best_violation and 
                     ((self.problem.minimize and current_fitness < best_fitness) or
                      (not self.problem.minimize and current_fitness > best_fitness)))):
                    best_solution = population[idx].copy()
                    best_fitness = current_fitness
                    best_violation = current_violation
            
            best_per_gen.append(best_fitness)
            
            # Create mutants using DE/best/1 strategy
            best_idx = sorted_indices[0]  # Use best ranked solution for mutation
            mutants = self.mutation_best_1(population, best_idx)
            
            # Create trial vectors through binomial crossover
            trials = self.crossover_bin(population, mutants)
            
            # Evaluate trials
            trial_obj_values = np.array([self.problem.objective_function(ind) 
                                       for ind in trials])
            trial_violations = np.array([self.evaluate_constraints(ind)[0] 
                                       for ind in trials])
            
            # Combine current population and trials for ranking
            combined_pop = np.vstack((population, trials))
            combined_obj = np.concatenate((obj_values, trial_obj_values))
            combined_violations = np.concatenate((constraint_violations, 
                                               trial_violations))
            
            # Rank combined population
            ranked_indices = self.stochastic_ranking(combined_pop, combined_obj, 
                                                   combined_violations)
            
            # Select best population_size individuals
            population = combined_pop[ranked_indices[:self.population_size]]
            
        
        return best_solution, best_fitness, best_per_gen