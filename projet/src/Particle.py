import time
import numpy as np
import random
from MaxCoveringProblem import MaxCoveringProblem

class Parameters:
    """Classe pour encapsuler tous les paramètres configurables du PSO"""
    def __init__(self, num_particles=50, max_iterations=1000, strategy="random", 
                 inertia_type="fixed", inertia_value=0.7, neighborhood_size=None, 
                 c1=1.5, c2=1.5, dist_type="HD", selection_type="stochastic", 
                 mutate=False, mutation_rate=0.1, particle_type="probabilistic"):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.initialization_strategy = strategy
        self.inertia_type = inertia_type
        self.inertia_value = inertia_value
        self.neighborhood_size = neighborhood_size if neighborhood_size else num_particles
        self.c1 = c1
        self.c2 = c2
        self.dist_type = dist_type
        self.selection_type = selection_type
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.particle_type = particle_type  # Nouveau paramètre

class PSO:
    def __init__(self, problem: MaxCoveringProblem, params: Parameters):
        self.problem = problem
        self.params = params
        self.global_best_position = None
        self.global_best_score = 0
        self.particles = self.initialize_particles()

        print(f"Initial best particle fitness: {self.global_best_score}")

    def initialize_particles(self):
        particles = []
        for _ in range(self.params.num_particles):
            # Création des particules selon le type spécifié
            if self.params.particle_type == "probabilistic":
                particle = ParticleProbabilistic(self.problem, self.params.initialization_strategy)
            elif self.params.particle_type == "flipcount":
                particle = ParticleFlipCount(self.problem, self.params.initialization_strategy)
            elif self.params.particle_type == "rparticle":
                particle = RParticle(self.problem, self.params.initialization_strategy)
            else:  # default
                particle = Particle(self.problem, self.params.initialization_strategy)
                
            particles.append(particle)
            if particle.best_score > self.global_best_score:
                self.global_best_position = np.copy(particle.best_position)
                self.global_best_score = particle.best_score
        return particles
    
    def get_inertia(self, iteration):
        if self.params.inertia_type == 'fixed':
            return self.params.inertia_value
        elif self.params.inertia_type == 'linear':
            return 0.9 - (0.5 * (iteration / self.params.max_iterations))
        elif self.params.inertia_type == 'nonlinear':
            return 0.9 * np.exp(-2 * (iteration / self.params.max_iterations))
    
    def get_neighborhood_best(self):
        neighborhood = np.random.choice(self.particles, self.params.neighborhood_size, replace=False)
        best_neighbor = max(neighborhood, key=lambda p: p.best_score)
        return best_neighbor.best_position
    
    def optimize(self, verbose=False):
        for iteration in range(self.params.max_iterations):
            w = self.get_inertia(iteration)
            for i, particle in enumerate(self.particles):
                best_position = self.get_neighborhood_best() if self.params.neighborhood_size < self.params.num_particles else self.global_best_position
                particle.update_velocity(best_position, w, self.params.c1, self.params.c2, self.params.dist_type)
                if self.params.mutate:
                    particle.mutate_velocity(self.params.mutation_rate)
                particle.update_position(selection_type=self.params.selection_type)
                particle.update_pbest()
                
                if particle.best_score > self.global_best_score:
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_score = particle.best_score

                if self.global_best_score == self.problem.n:
                    if verbose:
                        print(f"EARLY STOPPING Iteration {iteration + 1}/{self.params.max_iterations} - Best Score: {self.global_best_score}")
                    return self.global_best_position, self.global_best_score
            
            if verbose:
                print(f"Iteration {iteration + 1}/{self.params.max_iterations} - Best Score: {self.global_best_score}")
        
        return self.global_best_position, self.global_best_score

class Particle:
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        self.problem = problem
        self.position = self.initialize_position(strategy)
        self.best_position = np.copy(self.position)
        self.best_score = self.evaluate()

    def initialize_position(self, strategy: str):
        position = np.zeros(self.problem.m, dtype=int)
        
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
        elif type == "wHD":
            return np.sum(np.logical_xor(self.position, best) * np.array([len(s) for s in self.problem.subsets]))
        elif type == "bit-wise":
            return best - self.position
        
    def update_position(self):
        pass

    def update_pbest(self):
        score = self.evaluate()
        if score > self.best_score:
            self.best_position = self.position.copy()
            self.best_score = score

class ParticleProbabilistic(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        super().__init__(problem, strategy)
        self.velocity = np.random.uniform(0, 1, size=problem.m)

    def transfer_function(self, type="sigmoid"):
        if type == "sigmoid":
            probabilities = 1/(1 + np.exp(-self.velocity))
        return probabilities
    
    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        r1, r2 = random.random(), random.random()       
        self.velocity = w * self.velocity + c1 * r1 * self.distance(self.best_position, type=dist_type) + c2 * r2 * self.distance(global_best, dist_type)

    def mutate_velocity(self, mutation_rate=0.3):
        mutation_mask = np.random.rand(self.problem.m) < mutation_rate
        self.velocity[mutation_mask] *= -1

    def enforce_constraint(self):
        one_indices = np.where(self.position == 1)[0]
        excess = len(one_indices) - self.problem.k
        if excess > 0:
            self.position[np.random.choice(one_indices, size=excess, replace=False)] = 0   

    def update_position(self, tf_type="sigmoid", selection_type="stochastic"):
        super().update_position()
        probs = self.transfer_function(tf_type)

        if selection_type == "stochastic":
            if probs.sum() > 0:
                selected_indices = np.random.choice(
                    np.arange(self.problem.m), size=self.problem.k, 
                    p=probs/probs.sum(), replace=False
                )
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

class ParticleFlipCount(Particle):
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        super().__init__(problem, strategy)
        self.velocity = np.zeros(problem.m)  # Vitesse représentant le nombre de flips

    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * self.distance(self.best_position, type=dist_type)
        social = c2 * r2 * self.distance(global_best, type=dist_type)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        # Convert velocity to flip probabilities
        flip_probs = 1 / (1 + np.exp(-self.velocity))
        
        # Apply flips
        for i in range(len(self.position)):
            if random.random() < flip_probs[i]:
                self.position[i] = 1 - self.position[i]  # Flip the bit
        
        # Repair solution if needed
        self.enforce_constraints()

    def enforce_constraints(self):
        """Ensure exactly k subsets are selected"""
        selected = np.where(self.position == 1)[0]
        if len(selected) > self.problem.k:
            # Randomly deselect excess subsets
            to_deselect = np.random.choice(selected, size=len(selected)-self.problem.k, replace=False)
            self.position[to_deselect] = 0
        elif len(selected) < self.problem.k:
            # Randomly select additional subsets
            unselected = np.where(self.position == 0)[0]
            to_select = np.random.choice(unselected, size=self.problem.k-len(selected), replace=False)
            self.position[to_select] = 1

class RParticle(Particle):
    """Exemple de classe supplémentaire pour démontrer l'extensibilité"""
    def __init__(self, problem: MaxCoveringProblem, strategy="random"):
        super().__init__(problem, strategy)
        self.velocity = np.random.normal(0, 1, size=problem.m)

    def update_velocity(self, global_best, w=0.2, c1=1.5, c2=1.5, dist_type="HD"):
        r1, r2 = random.random(), random.random()
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (global_best - self.position)

    def update_position(self):
        self.position = (self.position + self.velocity > 0).astype(int)
        self.enforce_constraints()

    def enforce_constraints(self):
        selected = np.where(self.position == 1)[0]
        if len(selected) != self.problem.k:
            self.position = self.initialize_position("random")

if __name__ == "__main__":
    filename = "../data/scpc2.txt"  
    problem = MaxCoveringProblem(filename)
    
    # Test avec différentes configurations
    configs = [
        Parameters(particle_type="probabilistic", num_particles=30, max_iterations=500),
        Parameters(particle_type="flipcount", num_particles=30, max_iterations=500),
        Parameters(particle_type="rparticle", num_particles=30, max_iterations=500)
    ]
    
    for config in configs:
        print(f"\nTesting {config.particle_type} particles...")
        start_time = time.time()
        swarm = PSO(problem, config)
        best_pos, best_score = swarm.optimize(verbose=True)
        print(f"Best score: {best_score}/{problem.n}")
        print(f"Time: {time.time()-start_time:.2f}s")
