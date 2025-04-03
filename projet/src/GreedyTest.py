from Particle import Particle
from MaxCoveringProblem import MaxCoveringProblem
import os, csv 

class Greedy:
    def __init__(self, dataset_folder, file_list):
        self.dataset_folder = dataset_folder
        self.file_list = file_list
        self.results = []

    def run_test(self):
        for filename in self.file_list:
            file_path = os.path.join(self.dataset_folder, filename)
            problem = MaxCoveringProblem(file_path)
            particle = Particle(problem=problem, strategy="greedy")

            self.results.append({"filename": filename, "n": problem.n, "m": problem.m, "k": problem.k, "fitness": particle.best_score})

    def save_results_to_csv(self, output_file="../stats/greedy.csv"):
        """Save the results of the greedy algorithm to a CSV file."""
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "filename", "m", "n", "k", "fitness"])
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {output_file}")



def main():
    dataset_folder = "../data"
    test_files = sorted([filename for filename in os.listdir(dataset_folder)])[::-1] 
    
    greedy = Greedy(dataset_folder, test_files)
    greedy.run_test()
    greedy.save_results_to_csv()

    
if __name__ == "__main__":
    main()