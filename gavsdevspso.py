import numpy as np
from scipy import stats
from GeneticAlgorithm import Problem, GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from problems import setup_layeb05_problem, setup_layeb10_problem, setup_layeb15_problem, setup_layeb18_problem
from tabulate import tabulate
from HelperFunctions import compare_algorithms, run_multiple_trials, printDetailedResults, writeDetailedResultsToFile


if __name__ == "__main__":
    problems = {
        'Layeb05': setup_layeb05_problem(),
        'Layeb10': setup_layeb10_problem(),
        'Layeb15': setup_layeb15_problem(),
        'Layeb18': setup_layeb18_problem()
    }

    results_table = []
    pso_results = {
        'Layeb05': [-6.894047412475802, -6.904400303418801, -6.586698083102171, -6.900392409721636, -6.904939550239135, -6.906516018563297, -6.881063376650335, -6.760248087994617, -6.90379394570893, -6.8888884005475175, -6.9070907253474845, -6.843829947107128, -6.907173448204623, -6.907038097940196, -6.807799575207902, -6.742676729842308, -6.899701689879673, -6.902440913975023, -6.628073717680476, -6.600567604299864, -6.876056433171653, -6.89805577083038, -6.876642781346192, -6.8108838042347255, -6.897763761636209, -6.60645459073223, -6.595167811374888, -6.90599314818716, -6.9020567953149, -6.881222480082304],
        'Layeb10': [1.0475104804159003, 3.825761519838248, 0.7225202099818024, 0.6947016467793199, 3.2492790327862324, 7.77487783261859, 0.5071778822785936, 0.7799960056228704, 1.3128392061440826, 0.14994214563359734, 0.592171586349973, 0.5644482480556896, 0.15643755061830356, 0.8600147830634152, 1.1267579292290457, 3.884220622007287, 2.653944541569491, 0.6916857152614909, 0.030963646290744693, 0.6727988559821716, 3.3082416555178593, 0.9595712156800192, 1.3586564500234384, 1.1101421862440957, 1.282545714957088, 0.2551977110420368, 0.20047151882485167, 0.047621677839394194, 0.47155851538174004, 2.716370038181995],
        'Layeb15': [0.0009703436456001135, 0.000858126760187039, 0.024655844868793242, 0.11378070347201663, 0.01173269721679071, 0.0004428845611933463, 0.0007912397021146989, 0.0270405483611027, 0.002106163140508821, 0.004214044817990814, 0.0030814696173396827, 0.04016457679418739, 0.005825423488869164, 0.0033535996683875524, 0.0006661551273918764, 0.021222936537408588, 0.0018567080849327677, 0.0024286271701992046, 0.0026957768391903114, 0.018178596921797352, 0.0012611321590504376, 0.0030809096674675462, 0.00024150634162378726, 0.0017314950307740773, 7.262013713404958e-05, 0.00056572361280538, 0.004466098565772958, 0.0020436068891447334, 0.003280715711330928, 0.0014053933799188512],
        'Layeb18': [-5.768844929951439, -5.676271576437022, -6.201578736935437, -6.590019723397551, -5.684612236085743, -6.079458183634003, -6.095291684217189, -6.453922557043916, -6.334024812349047, -6.263008945894875, -6.511358025547899, -5.863175029960636, -6.209235486424132, -6.755911572300776, -6.37152572023542, -6.8345222127605245, -6.674747727220326, -6.811971320791301, -5.928161431180879, -5.575961300323928, -6.350520217406821, -5.567289169560795, -6.51744964161486, -5.815467978350262, -6.115461575266329, -6.636642426430557, -6.036132381161825, -6.033700054395309, -5.839277174436596, -5.949062193223546]
    }
    
    for name, problem in problems.items():
        print(f"\n=== Comparing algorithms on {name} problem ===")
        print("\n=== Running Using Genetic Algorithm ===")
        if name == 'Layeb05':
            resultsGA, statsGA, objectiveValuesGA = run_multiple_trials(problem, GeneticAlgorithm, num_trials=30, plot_convergence=True)
        else:
            resultsGA, statsGA, objectiveValuesGA = run_multiple_trials(problem, GeneticAlgorithm, num_trials=30)

        print(objectiveValuesGA)

        # Write detailed results to file
        writeDetailedResultsToFile(resultsGA, statsGA, objectiveValuesGA, name, "Genetic Algorithm")

        print("\n=== Running Differential Evolution ===")
        if name == 'Layeb05':
            resultsDE, statsDE, objectiveValuesDE = run_multiple_trials(problem, DifferentialEvolution, num_trials=30, plot_convergence=True)
        else:
            resultsDE, statsDE, objectiveValuesDE = run_multiple_trials(problem, DifferentialEvolution, num_trials=30)
        
        print(objectiveValuesDE)
        # Write detailed results to file
        writeDetailedResultsToFile(resultsDE, statsDE, objectiveValuesDE, name, "Differential Evolution")

        print("\n=== Comparing Using Wilcoxon Rank-Sum Test ===")
        pso_results_current = pso_results[name]
        pso_mean = np.mean(pso_results_current)
        pso_std = np.std(pso_results_current)

        print("\n Comparison GA vs DE")
        resultsGADE = compare_algorithms(objectiveValuesGA, objectiveValuesDE)

        print("\n Comparison GA vs PSO")
        resultsGAPSO = compare_algorithms(objectiveValuesGA, pso_results_current)

        print("\n Comparison DE vs PSO")
        resultsDEPSO = compare_algorithms(objectiveValuesDE, pso_results_current)

        results_table.append([
            name,
            f"{statsGA['mean']:.6f} ± {statsGA['std']:.6f}",
            f"{statsDE['mean']:.6f} ± {statsDE['std']:.6f}",
            f"{pso_mean:.6f} ± {pso_std:.6f}",
            f"{resultsGADE['p_value']:.6f}",
            f"{resultsGADE['effect_size']:.3f}",
            f"{resultsGAPSO['p_value']:.6f}",
            f"{resultsGAPSO['effect_size']:.3f}",
            f"{resultsDEPSO['p_value']:.6f}",
            f"{resultsDEPSO['effect_size']:.3f}"
        ])
    
    # Print comparison table
    headers = ["Problem", "GA (Mean ± Std)", "DE (Mean ± Std)" , 
              "PSO (Mean ± Std)", "p-value GA vs DE", "Effect Size GA vs DE", 
              "p-value GA vs PSO", "Effect Size GA vs PSO",
              "p-value DE vs PSO", "Effect Size DE vs PSO"]
    print("\nComparison Results:")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))