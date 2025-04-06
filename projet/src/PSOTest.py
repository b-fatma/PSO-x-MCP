import csv
from MaxCoveringProblem import MaxCoveringProblem
from PSO import PSO
import os

class PSOTest:
    def __init__(self, dataset_folder, file_list, best_params, max_iterations=1000, repetitions=10, verbose=False):
        self.dataset_folder = dataset_folder
        self.file_list = file_list
        self.best_params = best_params
        self.best_score = 0
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.repetitions = repetitions
        self.results = []

    def run_pso(self):
        """Run PSO algorithm on all problem instances from the file list."""
        for filename in self.file_list:
            file_path = os.path.join(self.dataset_folder, filename)
            print(f"\nRunning on {filename}...")
            problem = MaxCoveringProblem(file_path)

            # Unpack the best parameters
            params = self.best_params

            # Run PSO for 10 repetitions
            best_repetition_score = float('-inf')
            best_repetition_execution_time = 0
            for rep in range(self.repetitions):
                pso = PSO(problem, max_iterations=self.max_iterations, checkpoints=[1000], **params)
                _, best_score, scores = pso.optimize(self.verbose)

                assert len(scores) , 1

                # Track the best fitness across repetitions
                if best_score > best_repetition_score:
                    best_repetition_score = best_score
                    best_repetition_execution_time = scores[0]["execution_time(s)"]

            # Save the best result of this file
            self.results.append({
                "filename": filename, 
                "m": problem.m, 
                "n": problem.n, 
                "k": problem.k,
                "fitness": best_repetition_score, 
                "execution_time(s)": best_repetition_execution_time,
                **params
            })
            print(f"Tested {params} - Best Fitness: {best_repetition_score} \n")

        print(f"Best Score: {self.best_score}")

    def save_results_to_csv(self, output_file="pso_results.csv"):
        """Save the results of the PSO runs to a CSV file."""
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "filename", "m", "n", "k", "fitness", "execution_time(s)"] + list(self.best_params.keys()))
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {output_file}")



if __name__ == "__main__":

    # The best parameters and hyperparameters obtained after tuning
    best_param = {
        # Swarm parameters
        'num_particles': 100,
        'neighborhood_size': 30,
        # Particle parameters
        'inertia_type': 'fixed',
        'inertia_value': 0.7, 
        'c1': 1.5,
        'c2': 2.0,
        # Particle type parameters
        'mutate': True,
        'mutation_rate': 0.8,
        'dist_type': 'bit-wise', 
        'selection_type': 'stochastic'
    }

    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder) if filename.startswith("scpa")])[::-1] 

    pso_test = PSOTest(dataset_folder, test_files, best_param)
    pso_test.run_pso()

    pso_test.save_results_to_csv("../stats/pso_results_a2.csv")
