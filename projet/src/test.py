from MaxCoveringProblem import MaxCoveringProblem
from Particle import Particle, ParticleFlipCount, ParticleProbabilistic
import numpy as np

# Load the problem instance
filename = "../data/scp41.txt"
problem = MaxCoveringProblem(filename)

# Create two particles with different strategies
particle1 = Particle(problem, strategy="random")
particle2 = Particle(problem, strategy="random")


def test_difference(particle1, particle2):
    hd = particle1.distance(particle2.position, type="HD")
    norm_hd = particle1.distance(particle2.position, type="norm-HD")
    whd = particle1.distance(particle2.position, type="wHD")
    norm_whd = particle1.distance(particle2.position, type="norm-wHD")
    
    print(f"Hamming Distance: {hd}")
    print(f"Normalized Hamming Distance: {norm_hd:.4f}")
    print(f"Weighted Hamming Distance: {whd}")
    print(f"Normalized Weighted Hamming Distance: {norm_whd:.4f}")


def test_velocity_update(particle1, global_best):
    print("Before velocity update:")
    print(f"Velocity: {particle1.velocity}")
    
    particle1.update_velocity(global_best.position)
    
    print("After velocity update:")
    print(f"Velocity: {particle1.velocity}")

def test_particle_flipcount():
    particle = ParticleFlipCount(problem, strategy="random")
    global_best = ParticleFlipCount(problem, strategy="random")
    
    print("\nTesting ParticleFlipCount with multiple updates:")
    for i in range(5):  # Run multiple updates
        print(f"\nIteration {i+1}:")
        print(f"Position before update: {particle.position}")
        print(f"Velocity before update: {particle.velocity}")
        
        particle.update_velocity(global_best.position)
        print(f"Velocity after update: {particle.velocity}")
        
        particle.update_position()
        print(f"Position after update: {particle.position}")



def test_particle_probabilistic():
    particle = ParticleProbabilistic(problem, strategy="random")
    global_best = ParticleProbabilistic(problem, strategy="random")
    
    print("\nTesting ParticleProbabilistic with multiple updates:")
    for i in range(100):  # Run multiple updates
        # global_best = ParticleProbabilistic(problem, strategy="random")
        print(f"\nIteration {i+1}:")
        # print(f"Position before update: {particle.position}")
        # print(f"Velocity before update: {particle.velocity[:10]}")
        old_position = particle.position.copy()
        particle.update_velocity(global_best.position, dist_type="norm-wHD")
        print(f"Velocity after update: {particle.velocity[:10]}")
        
        particle.update_position(selection_type="stochastic")

        print(f"Position after update: {particle.position}")
        print(f"Fitness: {particle.evaluate()}, Best fitness: {particle.best_score}")
        print(f"HD between old and updated position: ", particle.distance(old_position))



# Run tests
test_difference(particle1, particle2)

global_best = Particle(problem, strategy="random")  # Assign a global best particle
test_velocity_update(particle1, global_best)

# test_particle_flipcount()

test_particle_probabilistic()