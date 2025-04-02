from MaxCoveringProblem import MaxCoveringProblem
import os
import numpy as np
import random
from math import ceil

class Particle:
    def __init__(self, problem: MaxCoveringProblem, strategy="random", particle_type="standard"):
        self.problem = problem
        self.particle_type = particle_type
        self.position = self.initialize_position(strategy)
        self.velocity = np.random.uniform(-1, 1, size=problem.m)
        self.best_position = np.copy(self.position)
        self.best_score = self.evaluate()

    def initialize_position(self, strategy: str):
        position = np.zeros(self.problem.m, dtype=int)
        
        if self.particle_type == "flip":
            return ParticleFlipCount(self.problem, strategy).position
        elif self.particle_type == "probabilistic":
            return ParticleProbabilistic(self.problem, strategy).position

        if strategy == "random":
            selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
        elif strategy == "greedy":
            selected_indices = np.argsort([-len(s) for s in self.problem.subsets])[:self.problem.k]
        elif strategy == "probabilistic":
            coverage_scores = np.array([len(s) for s in self.problem.subsets])  
            probabilities = coverage_scores / coverage_scores.sum()
            selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False, p=probabilities)
        elif strategy == "random-greedy":
            if random.random() < 0.5:
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
            else:
                selected_indices = np.argsort([-len(s) for s in self.problem.subsets])[:self.problem.k]
        elif strategy == "random-probabilistic":
            if random.random() < 0.5:
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False)
            else:
                coverage_scores = np.array([len(s) for s in self.problem.subsets])  
                probabilities = coverage_scores / coverage_scores.sum()
                selected_indices = np.random.choice(self.problem.m, size=self.problem.k, replace=False, p=probabilities)
        
        position[selected_indices] = 1
        return position

    def evaluate(self):
        covered_elements = set().union(*[self.problem.subsets[i] for i in range(self.problem.m) if self.position[i] == 1])
        return len(covered_elements)

    def distance(self, best, best_score=0, type="HD"):
        if type == "HD":
            return np.sum(np.logical_xor(self.position, best))
        elif type == "norm-HD":
            return np.sum(np.logical_xor(self.position, best)) / self.problem.m
        elif type == "wHD":
            return np.sum(np.logical_xor(self.position, best) * np.array([len(s) for s in self.problem.subsets]))
        elif type == "norm-wHD":
            subset_weights = np.array([len(s) for s in self.problem.subsets])
            total_weight = np.sum(subset_weights)
            return np.sum(np.logical_xor(self.position, best) * subset_weights) / total_weight if total_weight > 0 else 0
        elif type == "fitness":
            return best_score - self.evaluate()
        elif type == "bit-wise":
            return best - self.position

    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):   
        r1, r2 = random.random(), random.random()
        self.velocity = w * self.velocity + c1 * r1 * self.distance(self.best_position, type=dist_type) + c2 * r2 * self.distance(global_best, dist_type)

    def update_position(self):
        pass

    def update_pbest(self):
        score = self.evaluate()
        if score > self.best_score:
            self.best_position = self.position.copy()
            self.best_score = score

class ParticleFlipCount(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        super().__init__(problem, strategy)
        self.velocity = 0

    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5, dist_type="HD"):
        super().update_velocity(global_best, w, c1, c2, dist_type)
        self.velocity = ceil(self.velocity) if self.velocity < self.problem.m else self.problem.m

    def update_position(self):
        selected_indices = np.random.choice(self.problem.m, size=self.velocity, replace=False)
        self.position[selected_indices] = 1 - self.position[selected_indices]

class ParticleProbabilistic(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        super().__init__(problem, strategy)
        self.velocity = np.random.uniform(0, 1, size=problem.m)

    def transfer_function(self, type="sigmoid"):
        if type == "sigmoid":
            return 1 / (1 + np.exp(-self.velocity))

    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        super().update_velocity(global_best, w, c1, c2, dist_type)

    def mutate_velocity(self, mutation_rate=0.3):
        mutation_mask = np.random.rand(self.problem.m) < mutation_rate
        self.velocity[mutation_mask] *= -1

    def enforce_constraint(self):
        one_indices = np.where(self.position == 1)[0]
        excess = len(one_indices) - self.problem.k
        if excess > 0:
            self.position[np.random.choice(one_indices, size=excess, replace=False)] = 0

    def update_position(self, tf_type="sigmoid", selection_type="stochastic"):
        probs = self.transfer_function(tf_type)
        if selection_type == "stochastic":
            if probs.sum() > 0:
                selected_indices = np.random.choice(np.arange(self.problem.m), size=self.problem.k, p=probs / probs.sum(), replace=False)
            else:
                selected_indices = np.random.choice(np.arange(self.problem.m), size=self.problem.k, replace=False)
            self.position[:] = 0
            self.position[selected_indices] = 1  
        elif selection_type == "deterministic":
            selected_indices = np.argsort(-probs)[:self.problem.k]
            self.position[:] = 0  
            self.position[selected_indices] = 1  
        elif selection_type == "standard":
            self.position = np.array([1 if random.random() < p else 0 for p in probs])
            self.enforce_constraint()
