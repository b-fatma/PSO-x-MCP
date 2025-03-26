from MaxCoveringProblem import MaxCoveringProblem
import os
import numpy as np
import random

class Particle:
    def __init__(self, problem: MaxCoveringProblem, strategy = "random"):
        self.problem = problem
        self.position = self.initialize_position(strategy)
        # VELOCITY INITIALIZATION NOT FINAL, DEPENDS ON WETHER PROBABLISTIC OR HAMMING DISTANCE PSO IS USED
        self.velocity = np.random.uniform(-1, 1, size=problem.m)
        self.best_position = np.copy(self.position)
        self.best_score = self.evaluate()

    def initialize_position(self, strategy: str):
        position = np.zeros(self.problem.m, dtype=int)

        # Randomly select exactly k subsets
        if strategy == "random":
            selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
        
        # Select k subsets that cover the most uncovered elements, 
        # the underlying assumption is that subsets with more elements, combined, cover more elements
        # it is important to note that this assumption is unwarranted 
        elif strategy == "greedy":
            selected_indices = np.argsort([-len(s) for s in self.problem.subsets])[:self.problem.k]

        # Coverage-based Probabilistic Selection
        # Selects k subsets with a probability proportional to their coverage.
        elif strategy == "probabilistic":
            coverage_scores = np.array([len(s) for s in problem.subsets])  
            probabilities = coverage_scores / coverage_scores.sum()  # Normalize to create probabilities
            selected_indices = np.random.choice(problem.m, size=self.problem.k, replace=False, p=probabilities)

        # Hybrid selection that combines random and greedy
        elif strategy == "random-greedy":
            if random.random() < 0.5:
                # Random
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
            else:
                # Greedy
                selected_indices = np.argsort([-len(s) for s in self.problem.subsets])[:self.problem.k]

        elif strategy == "random-probabilistic":
            if random.random() < 0.5:
                # Random
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
            else:
                # Coverage-based Probabilistic Selection
                coverage_scores = np.array([len(s) for s in problem.subsets])  
                probabilities = coverage_scores / coverage_scores.sum()  # Normalize to create probabilities
                selected_indices = np.random.choice(problem.m, size=self.problem.k, replace=False, p=probabilities)
  
        position[selected_indices] = 1

        return position
    
    def evaluate(self):
        # Calculate the number of unique covered elements for the current position
        covered_elements = set().union(*[self.problem.subsets[i] for i in range(self.problem.m) if self.position[i] == 1])
        # covered_elements = set()
        # for i in range(self.problem.m):
        #     if self.position[i] == 1:
        #         covered_elements.update(self.problem.subsets[i])
        return len(covered_elements) 
    
    # TO BE IMPLEMENTED, DEPENDS ON WETHER PROBABLISTIC OR HAMMING DISTANCE PSO IS USED
    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
        pass
        
    # TO BE IMPLEMENTED, DEPENDS ON WETHER PROBABLISTIC OR HAMMING DISTANCE PSO IS USED
    def update_position(self):
        pass
        

# if __name__ == "__main__":
#     strategies = ["greedy", "random", "probabilistic", "random-greedy", "random-probabilistic"]
#     dir = "../data/"
#     for filename in os.listdir(dir):
#         print(filename)
#         problem  = MaxCoveringProblem(dir + filename)
#         print(filename, problem.m, problem.n, len(problem.subsets), problem.k)
#         for strategy in strategies:
#             particle = Particle(problem, strategy)
#             print(strategy, particle.best_score)

