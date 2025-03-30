from MaxCoveringProblem import MaxCoveringProblem
import os
import numpy as np
import random
from math import ceil

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
            coverage_scores = np.array([len(s) for s in self.problem.subsets])  
            probabilities = coverage_scores / coverage_scores.sum()  # Normalize to create probabilities
            selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False, p=probabilities)

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
                coverage_scores = np.array([len(s) for s in self.problem.subsets])  
                probabilities = coverage_scores / coverage_scores.sum()  # Normalize to create probabilities
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False, p=probabilities)
  
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
    
    
    def distance(self, best, best_score = 0, type="HD"):
            """
            Bitwise XOR (Standard in BPSO) aka Hamming distance
            OR Normalized hamming distance (we divide by /m), helps in velocity scaling
                (x−pbest)=sum(x xor pbest)
                Used in standard Binary PSO (Kennedy & Eberhart, 1997).

            Set-Based Difference (for Covering Problems) | it is computationally expensive 
                (x−pbest)={subsets in pbest but not in x}

                Instead of treating x and pbest​ as bit vectors, treat them as sets of selected elements.

                This is useful for set-based optimization problems like MCP because it explicitly tells which elements are missing rather than just counting mismatches.

            Weighted Hamming Distance
            (We can normalized wHD as well) 
                wDH=sum(wi(xi xor pbest,i))

                Assign different weights to different bits based on their importance (the coverage contribution of a subset).

                Useful for MCP where some subsets cover more elements than others.
            """
            if type == "HD":
                return np.sum(np.logical_xor(self.position, best))
            
            elif type == "norm-HD":
                return np.sum(np.logical_xor(self.position, best)) / self.problem.m

            elif type == "wHD":
                return np.sum(np.logical_xor(self.position, best) * np.array([len(s) for s in self.problem.subsets]))
            
            elif type  == "norm-wHD":
                subset_weights = np.array([len(s) for s in self.problem.subsets])
                total_weight = np.sum(subset_weights)
                return np.sum(np.logical_xor(self.position, best) * subset_weights) / total_weight if total_weight > 0 else 0
            
            elif type == "fitness":
                return best_score - self.evaluate()
            
            elif type == "bit-wise":
                return best- self.position

    # TO BE IMPLEMENTED, DEPENDS ON WETHER PROBABLISTIC OR HAMMING DISTANCE PSO IS USED
    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):   
        r1, r2 = random.random(), random.random()       
        self.velocity = w * self.velocity + c1 * r1 * self.distance(self.best_position, type=dist_type) + c2 * r2 * self.distance(global_best, dist_type)

        
    # TO BE IMPLEMENTED, DEPENDS ON WETHER PROBABLISTIC OR HAMMING DISTANCE PSO IS USED
    def update_position(self):
        pass

    def update_pbest(self):
        score = self.evaluate()
        if score > self.best_score:
            self.best_position = self.position.copy()
            self.best_score = score

# The problem with this implementation is :
#   - We cannot control the number of selected subsets (bits set to 1 in self.position)
#   - Velocity saturates at problem.m
class ParticleFlipCount(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy = "random"):
        super().__init__(problem, strategy)
        self.velocity = 0

    # Velocity converges to m very fast using this formula, needs improvement
    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5, dist_type="HD"):
        super().update_velocity(global_best, w, c1, c2, dist_type)
        self.velocity = ceil(self.velocity) if self.velocity < self.problem.m else self.problem.m

    def update_position(self):
        super().update_position()
        selected_indices = np.random.choice(self.problem.m, size=self.velocity, replace=False)
        self.position[selected_indices] = 1 - self.position[selected_indices]



class ParticleProbabilistic(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy = "random"):
        super().__init__(problem, strategy)
        self.velocity = np.random.uniform(0, 1, size=problem.m)

    def transfer_function(self, type="sigmoid"):
        if type == "sigmoid":
            probabilities = 1/(1 + np.exp(-self.velocity))
        return probabilities
    
    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        super().update_velocity(global_best, w, c1, c2, dist_type)
        # Min max scaling (mean 0, std dev 8), to prevent velocities from overshooting
        # self.velocity = 8 * (self.velocity - np.min(self.velocity)) / (np.max(self.velocity) - np.min(self.velocity)) - 4

    # Based on the paper :  https://www.researchgate.net/publication/31208097_Binary_Particle_Swarm_Optimization_with_Bit_Change_Mutation
    def mutate_velocity(self, mutation_rate=0.3):
        mutation_mask = np.random.rand(self.problem.m) < mutation_rate
        self.velocity[mutation_mask] *= -1

     
    def update_position(self, tf_type="sigmoid", selection_type="stochastic"):
        super().update_position()
        probs = self.transfer_function(tf_type)

        if selection_type == "stochastic":
            # Stochastic selection of k positions based on probabilities
            selected_indices = np.random.choice(np.arange(self.problem.m), size=self.problem.k, p=probs/probs.sum(), replace=False)
        else:
            # Select k best velocities: initial experiments show that this leads to early convergence because of lack of exploration
            selected_indices = np.argsort(-probs)[:self.problem.k]

        self.position[:] = 0
        self.position[selected_indices] = 1
        # self.position = np.array([1 if random.random() < p else 0 for p in probs])




if __name__ == "__main__":
    # strategies = ["greedy", "random", "probabilistic", "random-greedy", "random-probabilistic"]
    strategies = ["greedy", "random"]
    dir = "../data/"
    for filename in os.listdir(dir):
        print(filename)
        problem  = MaxCoveringProblem(dir + filename)
        print(filename, problem.m, problem.n, len(problem.subsets), problem.k)
        for strategy in strategies:
            # for _ in range(100):
                particle = Particle(problem, strategy)
                # if particle.best_score != problem.n:
                #     print("Non perfect position")
                print(strategy, particle.best_score, problem.n, particle.best_score / problem.n)
            # print([problem.subsets[i] for i in range(problem.m) if particle.position[i] == 1])
        # break

    # filename = "scp41.txt"
    # for _ in range(5):
    #     problem  = MaxCoveringProblem(dir + filename)
    #     particle = Particle(problem, "random")
    #     print("Randomly selected subsets:", particle.position, particle.best_score)


    # if __name__ == "__main__":
#     filename = "../testscp.txt"

#     problem  = MaxCoveringProblem(filename)
#     print(f"filename {filename}, m {problem.m}, n {problem.n}, subsets size = m {len(problem.subsets)}, subsets {problem.subsets}, k {problem.k}")
#     print(max([len(subset) for subset in problem.subsets]))
#     print(min([len(subset) for subset in problem.subsets]))
#     print(len(np.unique(np.concatenate([list(subset) for subset in problem.subsets]))))


