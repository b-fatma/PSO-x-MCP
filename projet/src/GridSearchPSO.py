from MaxCoveringProblem import MaxCoveringProblem
from PSO import PSO
import itertools

class GridSearchPSO:
    def __init__(self, problem: MaxCoveringProblem, param_grid, verbose=False):
        self.problem = problem
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = 0
        self.verbose = verbose

    def run_grid_search(self):
        keys, values = zip(*self.param_grid.items())
        for param_set in itertools.product(*values):
            params = dict(zip(keys, param_set))
            pso = PSO(self.problem, **params)
            _, score = pso.optimize(self.verbose)
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            print(f"Tested {params} - Score: {score}")
        print(f"Best Params: {self.best_params} - Best Score: {self.best_score}")


def main():
    filename = "../data/scp41.txt"  
    problem = MaxCoveringProblem(filename)

    # Define the parameter grid for grid search
    param_grid = {
        'num_particles': [30, 50, 100],
        'max_iterations': [100, 1000, 3000],
        'inertia_type': ['fixed', 'linear', 'nonlinear'],
        'inertia_value': [0.2, 0.5, 0.7],  # Only used if inertia_type is 'fixed'
        'neighborhood_size': [None, 10],
        'c1': [1.5],
        'c2': [1.5],
        'dist_type': ['HD', 'bit-wise', 'wHD'],
        'selection_type': ['stochastic', 'deterministic', 'standard']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    grid_search = GridSearchPSO(problem, param_grid)
    grid_search.run_grid_search()

if __name__ == "__main__":
    main()