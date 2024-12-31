import numpy as np
from typing import List, Callable, Dict, Tuple


class Problem:
    """Problem specification class"""
    def __init__(self, objective_function, constraints, bounds, name, minimize=True):
        self.objective_function = objective_function
        self.constraints = constraints
        self.bounds = bounds
        self.name = name
        self.minimize = minimize

class GeneticAlgorithm:
    def __init__(self, problem: Problem, 
                 population_size: int = 50,
                 num_generations: int = 200,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 plot_convergence: bool = False):
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
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
    
    def calculate_fitness(self, x, generation):
        """Calculate fitness with dynamic penalty"""
        obj_value = self.problem.objective_function(x)
        total_violation, _ = self.evaluate_constraints(x)
        
        # Dynamic penalty coefficient
        C = 2 * (generation / self.num_generations)
        # C = 1000 * (generation / self.num_generations)
        penalty = (C + 1) * (total_violation ** 2)
        
        if self.problem.minimize:
            return 1.0 / (obj_value + penalty + 1e-10)
        else:
            return obj_value - penalty
    
    def tournament_selection(self, population, fitness):
        """Binary tournament selection"""
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        if fitness[idx1] > fitness[idx2]:
            return population[idx1].copy()
        return population[idx2].copy()
    
    def crossover(self, parent1, parent2, eta_c=20):
        """Simulated Binary Crossover (SBX)"""
        if np.random.rand() <= self.crossover_prob:
            child1, child2 = np.copy(parent1), np.copy(parent2)
            for i in range(self.num_variables):
                if np.random.rand() <= 0.5:
                    beta = (2 * np.random.rand()) ** (1 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                    
                    # Ensure bounds
                    lower, upper = self.problem.bounds[i]
                    child1[i] = np.clip(child1[i], lower, upper)
                    child2[i] = np.clip(child2[i], lower, upper)
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutation(self, individual, eta_m=20):
        """Parameter-based mutation"""
        for i in range(self.num_variables):
            if np.random.rand() <= self.mutation_prob:
                lower, upper = self.problem.bounds[i]
                delta = min(individual[i] - lower, upper - individual[i]) / (upper - lower)
                
                u = np.random.uniform(0, 1)
                if u <= 0.5:
                    delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1))
                
                individual[i] = individual[i] + delta_q * (upper - lower)
                individual[i] = np.clip(individual[i], lower, upper)
        
        return individual
    
    def run(self):
        """Run the genetic algorithm"""
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('-inf')
        best_per_gen = []
        best_objective_per_gen = []
        
        for generation in range(self.num_generations):
            # Evaluate population
            fitness = np.array([self.calculate_fitness(ind, generation) for ind in population])
            
            # Track best solution
            gen_best_idx = np.argmax(fitness)
            gen_best = population[gen_best_idx]
            gen_best_fitness = fitness[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = gen_best.copy()
            
            best_per_gen.append(gen_best_fitness)
            best_objective_per_gen.append(self.problem.objective_function(gen_best))
            
            # Create new population
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population)
            
            
        
        return best_solution, self.problem.objective_function(best_solution), best_objective_per_gen