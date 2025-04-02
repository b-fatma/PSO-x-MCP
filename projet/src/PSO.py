import time
from MaxCoveringProblem import MaxCoveringProblem
from Particle import ParticleFlipCount, ParticleProbabilistic
import numpy as np
import csv

class PSO:
    def __init__(self, problem: MaxCoveringProblem, num_particles=50, max_iterations=1000, strategy="random", inertia_type="fixed", inertia_value=0.7, neighborhood_size=None, c1=1.5, c2=1.5, dist_type="HD", selection_type="stochastic", mutate=False, mutation_rate=0.1, particle_type="probabilistic",checkpoints=None):
        self.problem = problem
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.initialization_strategy = strategy
        self.inertia_type = inertia_type
        self.inertia_value = inertia_value  # Tunable when inertia_type is 'fixed'
        self.neighborhood_size = neighborhood_size if neighborhood_size else num_particles  # Defaults to global best if None
        self.global_best_position = None
        self.global_best_score = 0
        
        self.particle_type = particle_type  # New parameter for particle type
        self.particles = self.initialize_particles()

        # PSO optimize parameters
        self.c1 = c1
        self.c2 = c2
        self.dist_type = dist_type
        self.selection_type = selection_type
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.checkpoints = checkpoints if checkpoints else None


    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            # Initialize particles based on the particle type
            if self.particle_type == "probabilistic":
                particle = ParticleProbabilistic(self.problem, self.initialization_strategy)
            elif self.particle_type == "flip_count":
                particle = ParticleFlipCount(self.problem, self.initialization_strategy)
            else:
                raise ValueError(f"Unknown particle type: {self.particle_type}")
            
            particles.append(particle)
            if particle.best_score > self.global_best_score:
                self.global_best_position = np.copy(particle.best_position)
                self.global_best_score = particle.best_score

        print(f"Initial best particle fitness: {self.global_best_score}")
        return particles
    
    def get_inertia(self, iteration):
        if self.inertia_type == 'fixed':
            return self.inertia_value
        elif self.inertia_type == 'linear':
            return 0.9 - (0.5 * (iteration / self.max_iterations))
        elif self.inertia_type == 'nonlinear':
            return 0.9 * np.exp(-2 * (iteration / self.max_iterations))
    
    def get_neighborhood_best(self):
        neighborhood = np.random.choice(self.particles, self.neighborhood_size, replace=False)
        best_neighbor = max(neighborhood, key=lambda p: p.best_score)
        return best_neighbor.best_position
    
    def optimize(self, verbose=False):
        checkpoints_scores = {}
        start_time = time.time()  # Start timer
        for iteration in range(self.max_iterations):
            w = self.get_inertia(iteration)
            for i, particle in enumerate(self.particles):
                best_position = self.get_neighborhood_best() if self.neighborhood_size < self.num_particles else self.global_best_position
                particle.update_velocity(best_position, w, self.c1, self.c2, self.dist_type)
                if self.mutate:
                    particle.mutate_velocity(self.mutation_rate)
                particle.update_position(selection_type=self.selection_type)
                particle.update_pbest()
                
                if particle.best_score > self.global_best_score:
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_score = particle.best_score

                if self.checkpoints is not None and iteration in self.checkpoints:
                    checkpoints_scores[iteration] = self.global_best_score

                if self.global_best_score == self.problem.n:
                    if verbose:
                        print(f"EARLY STOPPING Iteration {iteration + 1}/{self.max_iterations} - Best Score: {self.global_best_score}")
                    end_time = time.time()  # End timer
                    exec_time = end_time - start_time
                    print(f"Total execution time: {exec_time:.2f} seconds")
                    self.save_results_to_csv(iteration+1, checkpoints_scores, self.global_best_score, exec_time)
                    return self.global_best_position, self.global_best_score, iteration + 1, checkpoints_scores
                   
            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations} - Best Score: {self.global_best_score}")
        
        end_time = time.time()  # End timer after the full loop
        exec_time = end_time - start_time
        print(f"Total execution time: {exec_time:.2f} seconds")
        self.save_results_to_csv(iteration+1, checkpoints_scores, self.global_best_score, exec_time) 
        return self.global_best_position, self.global_best_score, iteration + 1, checkpoints_scores, exec_time

    
    def save_results_to_csv(self, checkpoints_scores, final_iteration, final_score, exec_time):
        filename = "stats/pso_results.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            # write all checkpoint results
            for iteration, score in checkpoints_scores.items():
                writer.writerow([iteration, score])
            #  write the final result with time and execution time
            writer.writerow([final_iteration, final_score, exec_time])

        print(f"Results saved to {filename}")
            
if __name__ == "__main__":
    filename = "data/scpc2.txt"  
    problem = MaxCoveringProblem(filename)
    swarm = PSO(problem, num_particles=30, neighborhood_size=30, inertia_type="linear", max_iterations=500, strategy="random", dist_type="bit-wise", selection_type="standard", particle_type="flip_count",checkpoints=[100,200])  # Add particle_type
    best_position, best_score , iter , checkpoints, tim = swarm.optimize(verbose=False)
    print("Best Position:", best_position)
    print("Best Score:", best_score)
    print("Iterations:", iter)
    print("Checkpoints:", checkpoints)
    print("n", problem.n)
