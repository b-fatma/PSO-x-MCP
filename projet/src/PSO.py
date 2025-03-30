from MaxCoveringProblem import MaxCoveringProblem
from Particle import ParticleFlipCount, ParticleProbabilistic
import numpy as np

class PSO:
    def __init__(self, problem: MaxCoveringProblem, num_particles=50, max_iterations=1000, strategy="random"):
        self.problem = problem
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.initialization_strategy = strategy
        self.global_best_position = None
        self.global_best_score = 0
        
        self.particles = self.initialize_particles()

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):

            particle = ParticleProbabilistic(self.problem, self.initialization_strategy)
            
            particles.append(particle)
            if particle.best_score > self.global_best_score:
                self.global_best_position = np.copy(particle.best_position)
                self.global_best_score = particle.best_score

        print(f"Initial best particle fitness: {self.global_best_score}")
        return particles

    def optimize(self, w=0.2, c1=1.5, c2=1.5, dist_type="HD", selection_type="stochastic", mutate=False, mutation_rate=0.1, verbose=False):
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2, dist_type)
                if mutate:
                    particle.mutate_velocity(mutation_rate)
                particle.update_position(selection_type=selection_type)
                particle.update_pbest()
                
                if particle.best_score > self.global_best_score:
                    # if verbose:
                        # print(f"Iteration {iteration + 1}/{self.max_iterations} - Best Score: {self.global_best_score}")
                    self.global_best_position = np.copy(particle.best_position)
                    self.global_best_score = particle.best_score
            
            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations} - Best Score: {self.global_best_score}")

            # print([particle.best_score for particle in self.particles])
            # print([particle.evaluate() for particle in self.particles])
            # print([particle.velocity[:5] for particle in self.particles])
        
        return self.global_best_position, self.global_best_score

if __name__ == "__main__":
    filename = "../data/scp41.txt"  
    problem = MaxCoveringProblem(filename)
    swarm = PSO(problem, num_particles=50, max_iterations=1000, strategy="random")
    best_position, best_score = swarm.optimize(dist_type="norm-wHD", selection_type="stochastic", verbose=True)
    print("Best Position:", best_position)
    print("Best Score:", best_score)
    print("n", problem.n)