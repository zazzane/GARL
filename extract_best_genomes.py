import os
import pickle
import neat
import sys

# Path to your config file used during training
config_path = 'config-feedforward'  # Adjust if your file has a different name

# Load config
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Function to extract best genome from a checkpoint
def extract_best_genome(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    print(f"Population has {len(pop.population)} genomes")
    
    # Filter out genomes with None fitness
    valid_genomes = [(gid, g) for gid, g in pop.population.items() if g.fitness is not None]
    
    if not valid_genomes:
        print("ERROR: No genomes with valid fitness found in the checkpoint!")
        print("Fitness values:")
        for gid, genome in list(pop.population.items())[:10]:
            print(f"  Genome {gid}: fitness = {genome.fitness}")
        sys.exit(1)
    
    print(f"Found {len(valid_genomes)} genomes with valid fitness values")
    
    # Find the best genome
    best_id, best_genome = max(valid_genomes, key=lambda x: x[1].fitness)
    print(f"Best genome ID: {best_id}, fitness: {best_genome.fitness}")
    
    return best_genome

# Find latest checkpoints - handle directory not found errors
try:
    scout_dir = 'scout_checkpoints'
    scout_checkpoints = sorted([f for f in os.listdir(scout_dir) if f.startswith('comp_checkpoint')])
    if not scout_checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {scout_dir}")
    latest_scout_checkpoint = os.path.join(scout_dir, scout_checkpoints[-1])
    print(f"Latest scout checkpoint: {latest_scout_checkpoint}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please check the directory path and ensure it contains checkpoint files.")
    sys.exit(1)

try:
    guard_dir = 'guard_checkpoints'
    guard_checkpoints = sorted([f for f in os.listdir(guard_dir) if f.startswith('comp_checkpoint')])
    if not guard_checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {guard_dir}")
    latest_guard_checkpoint = os.path.join(guard_dir, guard_checkpoints[-1])
    print(f"Latest guard checkpoint: {latest_guard_checkpoint}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please check the directory path and ensure it contains checkpoint files.")
    sys.exit(1)

# Extract best genomes
print("\nExtracting best scout genome...")
best_scout = extract_best_genome(latest_scout_checkpoint)

print("\nExtracting best guard genome...")
best_guard = extract_best_genome(latest_guard_checkpoint)

# Save as pkl files
os.makedirs('models', exist_ok=True)
scout_path = 'models/best_competitive_scout_genome.pkl'
guard_path = 'models/best_competitive_guard_genome.pkl'

with open(scout_path, 'wb') as f:
    pickle.dump(best_scout, f)

with open(guard_path, 'wb') as f:
    pickle.dump(best_guard, f)

print(f"\nBest scout genome saved to: {scout_path}")
print(f"Best guard genome saved to: {guard_path}")

# Print fitness values for verification
print(f"\nScout fitness: {best_scout.fitness}")
print(f"Guard fitness: {best_guard.fitness}")