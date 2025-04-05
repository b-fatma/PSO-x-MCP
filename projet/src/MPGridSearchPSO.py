import itertools
import time
import csv
import multiprocessing, os
from MaxCoveringProblem import MaxCoveringProblem
from PSO import PSO

class MPGridSearchPSO:
    def __init__(self, dataset_folder, file_list, param_grid, max_iterations=1000, checkpoints=[100, 300, 500, 700], verbose=False):
        self.dataset_folder = dataset_folder
        self.file_list = file_list  
        self.param_grid = param_grid  
        self.best_params = None
        self.best_score = 0
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.checkpoints = checkpoints
        self.results = []
    
    def run_single_instance(self, filename, verbose=False):
        """Run grid search on a single file instance (for parallel execution)."""
        file_path = os.path.join(self.dataset_folder, filename)
        print(f"\nRunning Grid Search on {filename}...")
        problem = MaxCoveringProblem(file_path)
        local_results = []
        local_best_score = 0
        local_best_params = None

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

            if best_score > local_best_score:
                local_best_score = best_score
                local_best_params = params

            if verbose:
                print(f"Tested {params} - Score: {best_score} \n")

            for score in scores:
                local_results.append({"filename": filename, "n": problem.n, "m": problem.m, "k": problem.k} | params | score)

        return local_best_score, local_best_params, local_results

    def run_grid_search(self):
        """Run grid search in parallel across multiple files."""
        # Number of cores = 8 in the used hardware 
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
            results = pool.map(self.run_single_instance, self.file_list)

        for best_score, best_params, file_results in results:
            self.results.extend(file_results)
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_params = best_params

        print(f"Best Params: {self.best_params} - Best Score: {self.best_score}")

    def save_results_to_csv(self, output_file="grid_search_results.csv"):
        """Save the results of the grid search to a CSV file."""
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "filename", "m", "n", "k", "initial_fitness", "fitness", "iteration", "execution_time(s)"] + list(self.param_grid.keys()))
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {output_file}")



# Already run
def run_GS_BPSO_HDBPSO():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder)])[::-1] 
    print(test_files)

    param_grid = {
        # Swarm parameters
        'num_particles': [10, 30, 100],
        'neighborhood_size': [None, 10, 30],
        # Particle parameters
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        # Particle type parameters
        'dist_type': ['bit-wise', 'HD'],
        'selection_type': ['standard']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = MPGridSearchPSO(dataset_folder, test_files, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/bpso_hdbpso.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")



# Already run
def run_GS_wHDBPSO():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder)])[::-1] 
    print(test_files)

    param_grid = {
        # Swarm parameters
        'num_particles': [10, 30, 100],
        'neighborhood_size': [None, 10, 30],
        # Particle parameters
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        # Particle type parameters
        'dist_type': ['wHD'],
        'selection_type': ['standard']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = MPGridSearchPSO(dataset_folder, test_files, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/whdbpso.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")


# done
def run_GS_stochastic_BPSO_HDBPSO():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder)])[::-1] 

    param_grid = {
        # Swarm parameters
        'num_particles': [10, 30, 100],
        'neighborhood_size': [None, 10, 30],
        # Particle parameters
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        # Particle type parameters
        'dist_type': ['bit-wise', 'HD'],
        'selection_type': ['stochastic']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = MPGridSearchPSO(dataset_folder, test_files, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/stochastic_bpso_hdbpso.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")



# done
def run_GS_stochastic_wHDBPSO():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder)])[::-1] 
    print(test_files)

    param_grid = {
        # Swarm parameters
        'num_particles': [10, 30, 100],
        'neighborhood_size': [None, 10, 30],
        # Particle parameters
        'inertia_type': ['linear', 'fixed'],
        'inertia_value': [0.2, 0.5, 0.7], # Only used if inertia_type is 'fixed'
        'c1': [1.5, 2.0],
        'c2': [1.5, 2.0],
        # Particle type parameters
        'dist_type': ['wHD'],
        'selection_type': ['stochastic']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = MPGridSearchPSO(dataset_folder, test_files, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/stochastic_whdbpsoa.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")


# done
def run_GS_mutation():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder) if filename.startswith("scpa") or filename.startswith("scpb") or filename.startswith("scpc")])[::-1] 
    
    print(test_files)

    param_grid = {
        # Swarm parameters
        'num_particles': [100],
        'neighborhood_size': [30],
        # Particle parameters
        'inertia_type': ['fixed'],
        'inertia_value': [0.5], # Only used if inertia_type is 'fixed'
        'c1': [1.5],
        'c2': [2.0],
        # Particle type parameters
        'mutate': [True],
        'mutation_rate': [0.1, 0.2, 0.4, 0.8],
        'dist_type': ['bit-wise'], 
        'selection_type': ['stochastic']
    }

    print(f"\nRunning Grid Search with optimize params: {param_grid}")
    start_time = time.time()
    grid_search = MPGridSearchPSO(dataset_folder, test_files, param_grid)
    grid_search.run_grid_search()
    grid_search.save_results_to_csv("../stats/mutation_HD.csv")
    print(f"Run took {time.time()-start_time:.2}s\n\n")



if __name__ == "__main__":
    run_GS_mutation()

