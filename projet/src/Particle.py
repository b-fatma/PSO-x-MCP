from MaxCoveringProblem import MaxCoveringProblem
import os
import numpy as np
import random

class Particle:
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        self.problem = problem
        self.position = self.initialize_position(strategy)
        self.best_position = np.copy(self.position)
        self.best_score = self.evaluate()

    def initialize_position(self, strategy: str):
        """We implemented several initialization strategies but we only discussed and experimented with random initialization
        because the other variants lead to premature convergence"""
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

        else:
            raise ValueError(f"Unknown particle initialization strategy: {strategy}")
  
        position[selected_indices] = 1

        return position
    
    def evaluate(self):
        # Calculate the number of unique covered elements for the current position
        covered_elements = set().union(*[self.problem.subsets[i] for i in range(self.problem.m) if self.position[i] == 1])
        return len(covered_elements) 
    
    
    def distance(self, best, type="HD"):       
        if type == "HD":
            return np.sum(np.logical_xor(self.position, best))
        
        elif type == "wHD":
            return np.sum(np.logical_xor(self.position, best) * np.array([len(s) for s in self.problem.subsets]))
        
        elif type == "bit-wise":
            return best - self.position
        
        else:
            raise ValueError(f"Unknown distance type: {type}")
            
        
    def update_position(self):
        pass

    def update_pbest(self):
        score = self.evaluate()
        if score > self.best_score:
            self.best_position = self.position.copy()
            self.best_score = score

    def enforce_constraint(self):
        one_indices = np.where(self.position == 1)[0]
        excess = len(one_indices) - self.problem.k

        if excess > 0:
            self.position[np.random.choice(one_indices, size=excess, replace=False)] = 0   
        else:
            missing = self.problem.k - len(one_indices)
            if missing > 0:
                zero_indices = np.where(self.position == 0)[0]
                self.position[np.random.choice(zero_indices, size=missing, replace=False)] = 1  



class ParticleProbabilistic(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy = "random"):
        super().__init__(problem, strategy)
        self.velocity = np.random.uniform(0, 1, size=problem.m)

    def transfer_function(self, type="sigmoid"):
        if type == "sigmoid":
            probabilities = 1/(1 + np.exp(-self.velocity))
        else:
            raise ValueError(f"Unknown transfer function type: {type}")
        return probabilities
    
    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        r1, r2 = random.random(), random.random()       
        self.velocity = w * self.velocity + c1 * r1 * self.distance(self.best_position, type=dist_type) + c2 * r2 * self.distance(global_best, type=dist_type)

    # Based on the paper :  https://www.researchgate.net/publication/31208097_Binary_Particle_Swarm_Optimization_with_Bit_Change_Mutation
    def mutate_velocity(self, mutation_rate=0.3):
        mutation_mask = np.random.rand(self.problem.m) < mutation_rate
        self.velocity[mutation_mask] *= -1


    def update_position(self, tf_type="sigmoid", selection_type="stochastic"):
        super().update_position()
        probs = self.transfer_function(tf_type)
        # probs = self.velocity

        if selection_type == "stochastic":
            if probs.sum() > 0:  # Prevent division by zero
                selected_indices = np.random.choice(
                    np.arange(self.problem.m), size=self.problem.k, 
                    p=probs / probs.sum(), replace=False
                )
            else:
                selected_indices = np.random.choice(np.arange(self.problem.m), size=self.problem.k, replace=False)

            self.position[:] = 0  # Reset position
            self.position[selected_indices] = 1  

        # This method was not used in our experiments because it leads to premature convergence
        elif selection_type == "deterministic":
            selected_indices = np.argsort(-probs)[:self.problem.k]
            self.position[:] = 0  
            self.position[selected_indices] = 1  

        elif selection_type == "standard":
            self.position = np.array([1 if random.random() < p else 0 for p in probs])
            print(f"Before: {sum(self.position)}")
            self.enforce_constraint()

        else:
            raise ValueError(f"Unknown selection type: {selection_type}")



# We implemented BRPSO but did not discuss it nor experiment with it
# It's based on the following paper: https://www.mdpi.com/2313-7673/8/2/266
class ParticleRestructured(Particle):
    def __init__(self, problem, strategy="random"):
        super().__init__(problem, strategy)

    def update_position(self, global_best, p):
        r1 = random.random()
        self.position = r1 * self.best_position + (1 - r1) * global_best + p
        print(self.position)
        self.position = (self.position > np.random.random(self.problem.m)).astype(int)
        print(f"Before: {sum(self.position)}")
        self.enforce_constraint()
