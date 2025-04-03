import itertools
import time
import csv
from MaxCoveringProblem import MaxCoveringProblem
from PSO import PSO

class GridSearchPSO:
    def __init__(self, file_list, param_grid, max_iterations=1000, checkpoints=[100, 300, 500, 700], verbose=False):
        self.file_list = file_list  
        self.param_grid = param_grid  
        self.best_params = None
        self.best_score = 0
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.checkpoints = checkpoints
        self.results = []

    def run_grid_search(self):
        """Run grid search on all problem instances from the file list."""
        for filename in self.file_list:
            print(f"\nRunning Grid Search on {filename}...")
            problem = MaxCoveringProblem(filename)  

            keys, values = zip(*self.param_grid.items())
            for param_set in itertools.product(*values):
                params = dict(zip(keys, param_set))

                # Avoid redundant runs based on num_particles & neighborhood_size (since None already means num_particles == neighborhood_size)
                if params["num_particles"] == params["neighborhood_size"]:
                    continue
                
                # Avoid redundant runs because inertia_value is only relevant when inertia_type is 'fixed'
                if params["inertia_type"] == "linear" and "inertia_value" in params:
                    del params["inertia_value"]

                pso = PSO(problem, max_iterations=self.max_iterations, checkpoints=self.checkpoints, **params)
                _, best_score, scores = pso.optimize(self.verbose)

                if best_score > self.best_score:
                    self.best_score = best_score
                    self.best_params = params

                print(f"Tested {params} - Score: {best_score} \n")
                
                for score in scores:
                    self.results.append({"filename": filename, "n": problem.n, "m": problem.m, "k": problem.k} | params | score)

        print(f"Best Params: {self.best_params} - Best Score: {self.best_score}")

    def save_results_to_csv(self, output_file="grid_search_results.csv"):
        """Save the results of the grid search to a CSV file."""
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "filename", "m", "n", "k", "fitness", "iteration", "execution_time(s)"] + list(self.param_grid.keys()))
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {output_file}")

def run_tests_BPSO_HDBPSO(filenames):
    # Define the parameter grid for grid search
    param_grid = {
        'num_particles': [10, 30, 100],
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'neighborhood_size': [None, 10, 30],
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        'dist_type': ['bit-wise', 'HD'],
        'selection_type': ['standard']
    }

    grid_search = GridSearchPSO(filenames, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/BPSO_HDBPSO.csv")

def main():
    filenames = ["../data/scp41.txt"]  

    # Define the parameter grid for grid search
    param_grid = {
        'num_particles': [10, 30, 100],
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'neighborhood_size': [None, 10, 30],
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        'dist_type': ['bit-wise', 'HD'],
        'selection_type': ['standard']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = GridSearchPSO(filenames, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/testgs.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")

if __name__ == "__main__":
    main()