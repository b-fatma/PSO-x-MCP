from MaxCoveringProblem import MaxCoveringProblem
from PSO import PSO
from DFS import DFS
from Particle import Particle

def run_algorithm(filename, algorithm):
    problem = MaxCoveringProblem(filename)

    if algorithm == "pso":
        params = {
            'num_particles': 100,
            'neighborhood_size': 30,
            'inertia_type': 'fixed',
            'inertia_value': 0.7,
            'c1': 1.5,
            'c2': 2.0,
            'mutate': True,
            'mutation_rate': 0.8,
            'dist_type': 'bit-wise',
            'selection_type': 'stochastic',
            'checkpoints' : [100]
        }
        pso = PSO(problem, **params)
        pso_solution, pso_score, scores = pso.optimize(verbose=False)
        return pso_score

    elif algorithm == "greedy":
        particle = Particle(problem=problem, strategy="greedy")
        return particle.best_score

    elif algorithm == "dfs":
        dfs_solver = DFS(problem, time_limit=30)
        best_score, best_selection, completed, execution_time = dfs_solver.solve_time_bound(verbose=False)
        return best_score

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def run_all_algorithms(filename, algorithms):
    results = {}
    for algorithm in algorithms:
        print(f"\nRunning {algorithm} on {filename}...")
        best_score = run_algorithm(filename, algorithm)
        coverage_rate = best_score /  MaxCoveringProblem(filename).n
        print(f"Algorithm: {algorithm}, Best Score: {best_score}")
        print(f"Coverage Rate: {coverage_rate:.2%}")
    return results

def main():
    file  = "scp41.txt"
    print(f"\nRunning algorithms on {file}...")
    run_all_algorithms(f"../data/{file}", ["pso","greedy", "dfs"])

if __name__ == "__main__":
    main()
