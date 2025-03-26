import os
import csv
from MaxCoveringProblem import MaxCoveringProblem
from DFS import DFS  

class DFSTest:
    def __init__(self, dataset_folder, time_limit=10, output_file="dfs_results.csv"):
        self.dataset_folder = dataset_folder  
        self.time_limit = time_limit  
        self.output_file = output_file
        self.results = []  


    def run_tests(self, file_list):
        for file_name in file_list:
            file_path = os.path.join(self.dataset_folder, file_name)
            print(f"\nRunning DFS on {file_name}...")

            problem = MaxCoveringProblem(file_path)
            dfs_solver = DFS(problem, time_limit=self.time_limit)

            best_fitness, best_selection, completed, execution_time = dfs_solver.solve_time_bound_(verbose=False)
            used_budget = sum(best_selection)
            valid = (used_budget == problem.k)  

            print(f"File: {file_name}, m={problem.m}, n={problem.n}, k={problem.k}, "
                f"Fitness: {best_fitness}, Used Budget: {used_budget}/{problem.k}, "
                f"Completed: {completed}, Valid: {valid}, Time: {execution_time:.2f} sec")

            self.results.append({
                "filename": file_name,
                "m": problem.m,
                "n": problem.n,
                "k": problem.k,
                "fitness": best_fitness,
                "used_budget": used_budget,
                "completed": completed,
                "valid": valid,
                "execution_time(s)": execution_time
            })


    def save_results_to_csv(self):
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "m", "n", "k", "fitness", "used_budget", "completed", "valid", "execution_time(s)"])
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {self.output_file}")


    def print_summary(self):
        print("\n===== DFS Test Summary =====")
        for result in self.results:
            print(f"{result['filename']} → Fitness: {result['fitness']}, Used Budget: {result['used_budget']}/{result['k']}, "
                  f"Completed: {result['completed']}, Valid: {result['valid']}")
            

if __name__ == "__main__":
    dataset_folder = "../data"
    output_file = "../stats/dfs_scp4.csv"
    time_limit = 3 * 60 * 60  # (3 hours)
    test_files = [filename for filename in os.listdir(dataset_folder) if filename.startswith("scp4")]  

    dfs_tester = DFSTest(dataset_folder, time_limit=time_limit, output_file=output_file)
    dfs_tester.run_tests(test_files)
    dfs_tester.print_summary()
    dfs_tester.save_results_to_csv()
