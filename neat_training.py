import os
import random
import pickle
import time
import numpy as np
import neat
import copy
from til_environment import gridworld
from feature_extraction import process_observation, get_action_from_network

class NeatTrainer:
    """Trainer for NEAT networks in the Scout vs Guards environment"""
    
    def __init__(self, config, num_generations=100, num_trials=5, max_steps=500):
        """
        Initialize the trainer
        
        Args:
            config: NEAT configuration
            num_generations: Number of generations to train
            num_trials: Number of trials per evaluation
            max_steps: Maximum steps per episode
        """
        self.config = config
        self.num_generations = num_generations
        self.num_trials = num_trials
        self.max_steps = max_steps
        self.best_scout_fitness = -float('inf')
        self.best_guard_fitness = -float('inf')
        self.best_scout_genome = None
        self.best_guard_genome = None
        self.fitness_history = {'scout': [], 'guard': []}
    
    def evaluate_genomes_as_scout(self, genomes, config):
        """
        Evaluate a list of genomes as scouts
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            total_reward = 0
            
            for _ in range(self.num_trials):
                env = gridworld.env(
                    env_wrappers=[],
                    render_mode=None,
                    debug=True,
                    novice=False
                )
                env.reset(seed=random.randint(0, 10000))
                
                # Get all initial agent IDs
                all_agents = list(env.agents)
                scout_id = all_agents[0]  # Assuming scout is first agent
                guard_ids = all_agents[1:]  # Assuming other agents are guards
                
                # Create a simple policy for guards (random actions)
                guard_policy = lambda obs: random.randint(0, 4)
                
                trial_reward = 0
                done = False
                step_counter = 0
                
                while not done and step_counter < self.max_steps:
                    step_counter += 1
                    
                    # Get current agent
                    current_agent = env.agent_selection
                    
                    # Get observation and info
                    observation, reward, termination, truncation, info = env.last()
                    
                    # Check if current agent is done
                    if termination or truncation:
                        # Agent is done, pass None as the action
                        env.step(None)
                        
                        # If scout is terminated, end the episode
                        if current_agent == scout_id:
                            trial_reward += reward  # Add final reward
                            done = True
                        continue
                    
                    # Take action based on agent type
                    if current_agent == scout_id:  # Scout agent
                        # Process observation for NEAT
                        inputs = process_observation(observation)
                        action = get_action_from_network(net, inputs)
                        trial_reward += reward
                    else:  # Guard agents
                        action = guard_policy(observation)
                    
                    env.step(action)
                    
                    # Check if all agents are done
                    if env.agents == []:
                        done = True
                
                total_reward += trial_reward
            
            # Average reward across trials
            fitness = total_reward / self.num_trials
            genome.fitness = fitness
            
            # Track best scout genome
            if fitness > self.best_scout_fitness:
                self.best_scout_fitness = fitness
                self.best_scout_genome = genome
                
    def evaluate_genomes_as_guard(self, genomes, config):
        """
        Evaluate a list of genomes as guards

        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            total_reward = 0

            for _ in range(self.num_trials):
                env = gridworld.env(
                    env_wrappers=[],
                    render_mode=None,
                    debug=True,
                    novice=False
                )
                env.reset(seed=random.randint(0, 10000))

                # Get all initial agent IDs
                initial_agents = list(env.agents)
                scout_id = initial_agents[0]  # Assuming scout is first agent
                guard_ids = initial_agents[1:]  # Assuming other agents are guards

                # Create a simpler, more predictable scout policy that's easier to catch
                def scout_policy(obs):
                    # Simple pattern-based movement to make it easier for guards to learn
                    step = obs['step']
                    direction = step % 4  # Cycle through directions
                    return direction

                # Add exploration bonus for guards
                guard_positions = {}
                trial_reward = 0
                done = False
                step_counter = 0

                # Track if guard captured scout (to provide bonus reward)
                scout_captured = False

                while not done and step_counter < self.max_steps:
                    step_counter += 1

                    # Check if there are any agents left
                    if len(env.agents) == 0:
                        # All agents are done
                        done = True
                        break

                    # Get current agent
                    current_agent = env.agent_selection

                    # Check if scout was captured (not in agents list anymore)
                    if scout_id not in env.agents and not scout_captured:
                        trial_reward += 50.0  # Big bonus for capturing scout
                        scout_captured = True

                    try:
                        # Get observation and info - guard this with try/except
                        observation, reward, termination, truncation, info = env.last()

                        # Check if current agent is done
                        if termination or truncation:
                            # Agent is done, pass None as the action
                            env.step(None)
                            continue

                        # Take action based on agent type
                        if current_agent == scout_id:  # Scout agent
                            action = scout_policy(observation)
                        else:  # Guard agents
                            # Store current position for exploration bonus
                            pos = tuple(observation['location'])
                            if current_agent not in guard_positions:
                                guard_positions[current_agent] = set()

                            # Process observation for NEAT
                            inputs = process_observation(observation)
                            action = get_action_from_network(net, inputs)

                            # Give small reward for exploring new positions
                            if pos not in guard_positions[current_agent]:
                                trial_reward += 0.1  # Small exploration bonus
                                guard_positions[current_agent].add(pos)

                            # Add the environment reward
                            trial_reward += reward

                            # Add proximity reward to encourage getting closer to scout
                            if scout_id in env.agents:  # Only if scout is still active
                                # Check if scout is visible in viewcone
                                viewcone = observation['viewcone']
                                for y in range(viewcone.shape[0]):
                                    for x in range(viewcone.shape[1]):
                                        if viewcone[y, x] & 0b100:  # Scout present bit
                                            # Scout is visible, add proximity reward
                                            proximity_reward = 2.0  # Fixed reward for seeing the scout
                                            trial_reward += proximity_reward

                        env.step(action)

                    except KeyError:
                        # Agent may have been removed, step the environment with None
                        env.step(None)

                        # If we get a KeyError and the scout is gone, this is likely a capture
                        if scout_id not in env.agents and not scout_captured:
                            trial_reward += 50.0  # Bonus for capturing scout
                            scout_captured = True

                    # Check if all agents are done
                    if len(env.agents) == 0:
                        done = True

                # Add terminal reward if scout was captured
                if scout_captured:
                    trial_reward += 10.0

                total_reward += trial_reward

            # Average reward across trials
            fitness = total_reward / self.num_trials
            genome.fitness = fitness

            # Track best guard genome
            if fitness > self.best_guard_fitness:
                self.best_guard_fitness = fitness
                self.best_guard_genome = genome

    def evaluate_competitive(self, scout_genome, guard_genome, config):
        """
        Evaluate scout and guard genomes against each other
        
        Args:
            scout_genome: Scout NEAT genome
            guard_genome: Guard NEAT genome
            config: NEAT configuration
            
        Returns:
            Tuple of (scout_fitness, guard_fitness)
        """
        scout_net = neat.nn.FeedForwardNetwork.create(scout_genome, config)
        guard_net = neat.nn.FeedForwardNetwork.create(guard_genome, config)
        
        scout_total_reward = 0
        guard_total_reward = 0
        
        for _ in range(self.num_trials):
            env = gridworld.env(
                env_wrappers=[],
                render_mode=None,
                debug=True,
                novice=False
            )
            env.reset(seed=random.randint(0, 10000))
            
            # Get all initial agent IDs
            all_agents = list(env.agents)
            scout_id = all_agents[0]  # Assuming scout is first agent
            guard_ids = all_agents[1:]  # Assuming other agents are guards
            
            scout_trial_reward = 0
            guard_trial_reward = 0
            done = False
            step_counter = 0
            
            while not done and step_counter < self.max_steps:
                step_counter += 1
                
                # Get current agent
                current_agent = env.agent_selection
                
                # Get observation and info
                observation, reward, termination, truncation, info = env.last()
                
                # Check if current agent is done
                if termination or truncation:
                    # Agent is done, pass None as the action
                    env.step(None)
                    
                    # Add final reward to appropriate agent
                    if current_agent == scout_id:
                        scout_trial_reward += reward
                    else:
                        guard_trial_reward += reward
                        
                    continue
                
                # Process observation for NEAT
                inputs = process_observation(observation)
                
                if current_agent == scout_id:  # Scout agent
                    action = get_action_from_network(scout_net, inputs)
                    scout_trial_reward += reward
                else:  # Guard agents
                    action = get_action_from_network(guard_net, inputs)
                    guard_trial_reward += reward
                
                env.step(action)
                
                # Check if all agents are done
                if env.agents == []:
                    done = True
            
            scout_total_reward += scout_trial_reward
            guard_total_reward += guard_trial_reward
        
        # Average rewards across trials
        scout_fitness = scout_total_reward / self.num_trials
        guard_fitness = guard_total_reward / self.num_trials
        
        return scout_fitness, guard_fitness

    def train_scout(self, checkpoint_prefix="scout_checkpoint"):
        """
        Train a NEAT population for the scout role
        
        Args:
            checkpoint_prefix: Prefix for checkpoint files
            
        Returns:
            Best genome
        """
        # Ensure directory for checkpoint exists
        os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)
        
        pop = neat.Population(self.config)
        
        # Add reporters
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))
        
        # Run for up to num_generations generations
        best_genome = pop.run(self.evaluate_genomes_as_scout, self.num_generations)
        
        # Save the best genome
        with open('best_scout_genome.pkl', 'wb') as f:
            pickle.dump(best_genome, f)
        
        return best_genome
    
    def train_guard(self, checkpoint_prefix="guard_checkpoint"):
        """
        Train a NEAT population for the guard role
        
        Args:
            checkpoint_prefix: Prefix for checkpoint files
            
        Returns:
            Best genome
        """
        # Ensure directory for checkpoint exists
        os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)
        
        pop = neat.Population(self.config)
        
        # Add reporters
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))
        
        # Run for up to num_generations generations
        best_genome = pop.run(self.evaluate_genomes_as_guard, self.num_generations)
        
        # Save the best genome
        with open('best_guard_genome.pkl', 'wb') as f:
            pickle.dump(best_genome, f)
        
        return best_genome
    
    def evaluate_genomes_competitive(self, scout_genomes, guard_genomes):
        """
        Evaluate scout and guard genomes competitively
        
        Args:
            scout_genomes: List of (genome_id, genome) tuples for scouts
            guard_genomes: List of (genome_id, genome) tuples for guards
        """
        # Get best guard for evaluating scouts
        if self.best_guard_genome:
            best_guard = self.best_guard_genome
        elif guard_genomes:
            best_guard = guard_genomes[0][1]  # Use first guard as default
        else:
            # Create a random guard if none available
            best_guard = neat.DefaultGenome(0)
            best_guard.configure_new(self.config.genome_config)
            
        # Get best scout for evaluating guards
        if self.best_scout_genome:
            best_scout = self.best_scout_genome
        elif scout_genomes:
            best_scout = scout_genomes[0][1]  # Use first scout as default
        else:
            # Create a random scout if none available
            best_scout = neat.DefaultGenome(0)
            best_scout.configure_new(self.config.genome_config)
            
        # Evaluate all scouts against best guard
        for scout_id, scout_genome in scout_genomes:
            scout_fitness, _ = self.evaluate_competitive(scout_genome, best_guard, self.config)
            scout_genome.fitness = scout_fitness
            
        # Update best scout
        current_best_scout = max([g for _, g in scout_genomes], key=lambda x: x.fitness)
        if current_best_scout.fitness > self.best_scout_fitness:
            self.best_scout_fitness = current_best_scout.fitness
            self.best_scout_genome = current_best_scout
            
        # Evaluate all guards against best scout
        for guard_id, guard_genome in guard_genomes:
            _, guard_fitness = self.evaluate_competitive(best_scout, guard_genome, self.config)
            guard_genome.fitness = guard_fitness
            
        # Update best guard
        current_best_guard = max([g for _, g in guard_genomes], key=lambda x: x.fitness)
        if current_best_guard.fitness > self.best_guard_fitness:
            self.best_guard_fitness = current_best_guard.fitness
            self.best_guard_genome = current_best_guard
    
    
    def train_competitive(self, scout_checkpoint=None, guard_checkpoint=None, 
                          checkpoint_prefix="competitive_checkpoint"):
        """
        Train scout and guard populations competitively with emergency reset capability

        Args:
            scout_checkpoint: Path to scout checkpoint file
            guard_checkpoint: Path to guard checkpoint file
            checkpoint_prefix: Prefix for checkpoint files

        Returns:
            Tuple of (best_scout_genome, best_guard_genome)
        """
        import copy  # Make sure this import is at the top of your file

        # Flag to track if emergency reset just happened
        guard_reset_flag = False

        # Ensure directories exist
        os.makedirs(os.path.dirname("scout_" + checkpoint_prefix), exist_ok=True)
        os.makedirs(os.path.dirname("guard_" + checkpoint_prefix), exist_ok=True)

        # Load or create populations
        if scout_checkpoint and os.path.exists(scout_checkpoint):
            scout_pop = neat.Checkpointer.restore_checkpoint(scout_checkpoint)
        else:
            scout_pop = neat.Population(self.config)

        if guard_checkpoint and os.path.exists(guard_checkpoint):
            guard_pop = neat.Checkpointer.restore_checkpoint(guard_checkpoint)
        else:
            guard_pop = neat.Population(self.config)

        # Add reporters
        scout_pop.add_reporter(neat.StdOutReporter(True))
        scout_stats = neat.StatisticsReporter()
        scout_pop.add_reporter(scout_stats)
        scout_pop.add_reporter(neat.Checkpointer(5, filename_prefix="scout_" + checkpoint_prefix))

        guard_pop.add_reporter(neat.StdOutReporter(True))
        guard_stats = neat.StatisticsReporter()
        guard_pop.add_reporter(guard_stats)
        guard_pop.add_reporter(neat.Checkpointer(5, filename_prefix="guard_" + checkpoint_prefix))

        # Early stopping variables
        stagnation_threshold = 10  # Number of generations without improvement
        last_improvement_gen = 0
        best_combined_fitness = self.best_scout_fitness + self.best_guard_fitness

        # Training loop
        for generation in range(self.num_generations):
            print(f"Generation {generation}")

            # Create temporary fitness functions that just assign the pre-computed fitness
            def scout_fitness_function(genomes, config):
                pass  # The fitness is already assigned below

            def guard_fitness_function(genomes, config):
                pass  # The fitness is already assigned below

            # Evaluate all scout genomes against best guard
            if self.best_guard_genome:
                best_guard = self.best_guard_genome
            else:
                best_guard = list(guard_pop.population.values())[0]

            for genome_id, scout_genome in scout_pop.population.items():
                # Use deterministic but varied seed based on ID and generation
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                scout_fitness = 0

                for trial in range(self.num_trials):
                    # Different seed for each trial but deterministic
                    trial_seed = seed_base + (trial * 1000)
                    sf, _ = self._evaluate_single_competitive_trial(
                        scout_genome, best_guard, self.config, trial_seed)
                    scout_fitness += sf

                # Average fitness across trials
                scout_genome.fitness = scout_fitness / self.num_trials

            # Update best scout
            current_best_scout = max(scout_pop.population.values(), key=lambda x: x.fitness)
            if current_best_scout.fitness > self.best_scout_fitness:
                self.best_scout_fitness = current_best_scout.fitness
                self.best_scout_genome = current_best_scout
                with open('best_scout_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_scout_genome, f)

            # Evaluate all guard genomes against best scout
            if self.best_scout_genome:
                best_scout = self.best_scout_genome
            else:
                best_scout = list(scout_pop.population.values())[0]

            for genome_id, guard_genome in guard_pop.population.items():
                # Use deterministic but varied seed based on ID and generation
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                guard_fitness = 0

                for trial in range(self.num_trials):
                    # Different seed for each trial but deterministic
                    trial_seed = seed_base + (trial * 1000)
                    _, gf = self._evaluate_single_competitive_trial(
                        best_scout, guard_genome, self.config, trial_seed)
                    guard_fitness += gf

                # Average fitness across trials
                guard_genome.fitness = guard_fitness / self.num_trials

            # Update best guard
            current_best_guard = max(guard_pop.population.values(), key=lambda x: x.fitness)
            if current_best_guard.fitness > self.best_guard_fitness:
                self.best_guard_fitness = current_best_guard.fitness
                self.best_guard_genome = current_best_guard
                with open('best_guard_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_guard_genome, f)

            # Record fitness history
            self.fitness_history['scout'].append(self.best_scout_fitness)
            self.fitness_history['guard'].append(self.best_guard_fitness)

            # Check for improvement
            current_combined_fitness = self.best_scout_fitness + self.best_guard_fitness
            if current_combined_fitness > best_combined_fitness:
                print(f"Fitness improvement detected: {best_combined_fitness:.2f} -> {current_combined_fitness:.2f}")
                best_combined_fitness = current_combined_fitness
                last_improvement_gen = generation

            # ========== GUARD POPULATION EMERGENCY RESET (ROBUST SOLUTION) ==========
            # Check for critical stagnation in the guard population
            if not guard_reset_flag and hasattr(guard_pop.species, 'species') and guard_pop.species.species:
                stagnation_count = sum(1 for species in guard_pop.species.species.values() 
                                      if species.last_improved >= 10)
                total_species = len(guard_pop.species.species)

                # Execute emergency reset if 75% of species are stagnating and we have 5 or fewer species
                if total_species <= 5 and stagnation_count >= total_species * 0.75:
                    print("🚨 CRITICAL: Guard population requires emergency reset")
                    print(f"🚨 Current guard population size: {len(guard_pop.population)}")

                    # Save best genome
                    best_guard = max(guard_pop.population.values(), key=lambda x: x.fitness)
                    best_fitness = best_guard.fitness

                    # Save checkpoint objects for re-adding later
                    stdout_reporter = None
                    stats_reporter = None
                    checkpoint_reporter = None

                    # Extract individual reporter objects
                    for reporter in guard_pop.reporters.reporters:
                        if isinstance(reporter, neat.StdOutReporter):
                            stdout_reporter = reporter
                        elif isinstance(reporter, neat.StatisticsReporter):
                            stats_reporter = reporter
                        elif isinstance(reporter, neat.Checkpointer):
                            checkpoint_reporter = reporter

                    # Create a completely new population
                    guard_pop = neat.Population(self.config)

                    # Re-add the extracted reporters
                    if stdout_reporter:
                        guard_pop.add_reporter(stdout_reporter)
                    else:
                        guard_pop.add_reporter(neat.StdOutReporter(True))

                    if stats_reporter:
                        guard_pop.add_reporter(stats_reporter)
                    else:
                        guard_pop.add_reporter(neat.StatisticsReporter())

                    if checkpoint_reporter:
                        guard_pop.add_reporter(checkpoint_reporter)
                    else:
                        guard_pop.add_reporter(neat.Checkpointer(5, filename_prefix="guard_" + checkpoint_prefix))

                    print(f"✅ Created fresh population with {len(guard_pop.population)} members")

                    # Insert our best genome at a random position
                    if guard_pop.population:
                        random_id = next(iter(guard_pop.population))
                        guard_pop.population[random_id] = copy.deepcopy(best_guard)
                        print(f"✅ Inserted best genome (fitness {best_fitness:.2f}) at position {random_id}")

                    # Set the flag to skip run() call for this generation
                    guard_reset_flag = True

                    print("✅ Will skip guard.run() this generation to avoid errors")
                    print("✅ Emergency reset complete")

            # Early stopping check
            if generation - last_improvement_gen >= stagnation_threshold:
                print(f"Stopping early at generation {generation} due to {stagnation_threshold} generations without improvement")
                break

            # Create next generation using empty fitness functions (fitness is already assigned)
            scout_pop.run(scout_fitness_function, 1)

            # CRITICAL: Skip run() for guard population right after a reset
            if guard_reset_flag:
                print("⚠️ Skipping guard.run() after emergency reset")
                # Reset the flag for next generation
                guard_reset_flag = False
            else:
                guard_pop.run(guard_fitness_function, 1)

        return self.best_scout_genome, self.best_guard_genome

    def train_competitive_emergency(self, scout_checkpoint=None, guard_checkpoint=None, 
                                    checkpoint_prefix="checkpoints/comp_checkpoint"):
        """
        Emergency version of competitive training that completely rebuilds the guard population

        Args:
            scout_checkpoint: Path to scout checkpoint file
            guard_checkpoint: Path to guard checkpoint file
            checkpoint_prefix: Prefix for checkpoint files

        Returns:
            Tuple of (best_scout_genome, best_guard_genome)
        """
        import copy
        import os
        import random
        import pickle

        # Create directory structure
        os.makedirs(os.path.dirname("scout_" + checkpoint_prefix), exist_ok=True)
        os.makedirs(os.path.dirname("guard_" + checkpoint_prefix), exist_ok=True)

        # Load scout population
        if scout_checkpoint and os.path.exists(scout_checkpoint):
            print(f"Loading scout checkpoint from {scout_checkpoint}")
            scout_pop = neat.Checkpointer.restore_checkpoint(scout_checkpoint)
        else:
            print("Creating new scout population")
            scout_pop = neat.Population(self.config)

        # Add reporters to scout population
        scout_pop.add_reporter(neat.StdOutReporter(True))
        scout_stats = neat.StatisticsReporter()
        scout_pop.add_reporter(scout_stats)
        scout_pop.add_reporter(neat.Checkpointer(5, filename_prefix="scout_" + checkpoint_prefix))

        # Try to extract the best guard genome from the checkpoint or create a new one
        try:
            # First try to recover just the best genome
            if os.path.exists("best_guard_genome.pkl"):
                print("✅ Loading best guard genome from file")
                with open('best_guard_genome.pkl', 'rb') as f:
                    best_guard_genome = pickle.load(f)
                    self.best_guard_genome = best_guard_genome
                    self.best_guard_fitness = best_guard_genome.fitness
            elif guard_checkpoint and os.path.exists(guard_checkpoint):
                print(f"⚠️ Loading just the best genome from {guard_checkpoint}")
                try:
                    # Create a temporary population to extract the best genome
                    temp_pop = neat.Checkpointer.restore_checkpoint(guard_checkpoint)
                    if temp_pop.population:
                        # Extract the best genome
                        best_guard = max(temp_pop.population.values(), key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -float('inf'))
                        if hasattr(best_guard, 'fitness') and best_guard.fitness is not None:
                            self.best_guard_genome = best_guard
                            self.best_guard_fitness = best_guard.fitness
                            print(f"✅ Recovered best guard genome with fitness {self.best_guard_fitness}")
                except Exception as e:
                    print(f"⚠️ Error loading guard checkpoint: {str(e)}")
        except Exception as e:
            print(f"⚠️ Error recovering guard genome: {str(e)}")

        # Create a completely new guard population
        print("🔄 Creating fresh guard population from scratch")
        guard_pop = neat.Population(self.config)

        # Add reporters to guard population 
        guard_pop.add_reporter(neat.StdOutReporter(True))
        guard_stats = neat.StatisticsReporter()
        guard_pop.add_reporter(guard_stats)
        guard_pop.add_reporter(neat.Checkpointer(5, filename_prefix="guard_" + checkpoint_prefix))

        # Inject the best genome if we have one
        if self.best_guard_genome is not None:
            print("✅ Injecting best guard genome into new population")
            first_id = next(iter(guard_pop.population))
            guard_pop.population[first_id] = copy.deepcopy(self.best_guard_genome)

        # Early stopping variables
        stagnation_threshold = 10
        last_improvement_gen = 0
        best_combined_fitness = self.best_scout_fitness + self.best_guard_fitness

        # Training loop
        for generation in range(self.num_generations):
            print(f"\n======== Generation {generation} ========")

            # Create dummy fitness functions
            def scout_fitness_function(genomes, config):
                pass

            def guard_fitness_function(genomes, config):
                pass

            # Evaluate scout genomes against best guard
            if self.best_guard_genome:
                best_guard = self.best_guard_genome
            else:
                best_guard = list(guard_pop.population.values())[0]

            print(f"Evaluating {len(scout_pop.population)} scout genomes...")
            for genome_id, scout_genome in scout_pop.population.items():
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                scout_fitness = 0

                for trial in range(self.num_trials):
                    trial_seed = seed_base + (trial * 1000)
                    sf, _ = self._evaluate_single_competitive_trial(
                        scout_genome, best_guard, self.config, trial_seed)
                    scout_fitness += sf

                scout_genome.fitness = scout_fitness / self.num_trials

            # Update best scout
            current_best_scout = max(scout_pop.population.values(), key=lambda x: x.fitness)
            if current_best_scout.fitness > self.best_scout_fitness:
                print(f"📈 New best scout: {current_best_scout.fitness:.2f} (previous: {self.best_scout_fitness:.2f})")
                self.best_scout_fitness = current_best_scout.fitness
                self.best_scout_genome = current_best_scout
                with open('best_scout_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_scout_genome, f)

            # Evaluate guard genomes against best scout
            if self.best_scout_genome:
                best_scout = self.best_scout_genome
            else:
                best_scout = list(scout_pop.population.values())[0]

            print(f"Evaluating {len(guard_pop.population)} guard genomes...")
            for genome_id, guard_genome in guard_pop.population.items():
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                guard_fitness = 0

                for trial in range(self.num_trials):
                    trial_seed = seed_base + (trial * 1000)
                    _, gf = self._evaluate_single_competitive_trial(
                        best_scout, guard_genome, self.config, trial_seed)
                    guard_fitness += gf

                guard_genome.fitness = guard_fitness / self.num_trials

            # Update best guard
            current_best_guard = max(guard_pop.population.values(), key=lambda x: x.fitness)
            if current_best_guard.fitness > self.best_guard_fitness:
                print(f"📈 New best guard: {current_best_guard.fitness:.2f} (previous: {self.best_guard_fitness:.2f})")
                self.best_guard_fitness = current_best_guard.fitness
                self.best_guard_genome = current_best_guard
                with open('best_guard_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_guard_genome, f)

            # Record fitness history
            self.fitness_history['scout'].append(self.best_scout_fitness)
            self.fitness_history['guard'].append(self.best_guard_fitness)

            # Check for improvement
            current_combined_fitness = self.best_scout_fitness + self.best_guard_fitness
            if current_combined_fitness > best_combined_fitness:
                print(f"📊 Fitness improvement: {best_combined_fitness:.2f} -> {current_combined_fitness:.2f}")
                best_combined_fitness = current_combined_fitness
                last_improvement_gen = generation

            # Early stopping check
            if generation - last_improvement_gen >= stagnation_threshold:
                print(f"🛑 Stopping early at generation {generation} - no improvement for {stagnation_threshold} generations")
                break

            # Create next generations
            print("Evolving scout population...")
            scout_pop.run(scout_fitness_function, 1)

            print("Evolving guard population...")
            try:
                guard_pop.run(guard_fitness_function, 1)
            except Exception as e:
                print(f"⚠️ Error evolving guard population: {str(e)}")
                print("🔄 Creating fresh guard population...")

                # Save the best genome
                if self.best_guard_genome:
                    best_guard = self.best_guard_genome
                    best_fitness = self.best_guard_fitness
                else:
                    best_guard = max(guard_pop.population.values(), key=lambda x: x.fitness if hasattr(x, 'fitness') and x.fitness is not None else -float('inf'))
                    best_fitness = best_guard.fitness if hasattr(best_guard, 'fitness') and best_guard.fitness is not None else 0

                # Save reporters
                reporters = guard_pop.reporters

                # Create new population
                guard_pop = neat.Population(self.config)

                # Restore reporters
                for reporter in reporters:
                    guard_pop.add_reporter(reporter)

                # Insert best genome
                first_id = next(iter(guard_pop.population))
                guard_pop.population[first_id] = copy.deepcopy(best_guard)
                print(f"✅ Inserted best genome with fitness {best_fitness}")

        print("\n======== Training Complete ========")
        return self.best_scout_genome, self.best_guard_genome
    
    def train_competitive_ultimate(self, scout_checkpoint=None, guard_checkpoint=None, 
                                    checkpoint_prefix="checkpoints/comp_checkpoint"):
        """
        Ultimate emergency version of competitive training that completely rebuilds 
        the guard population and handles all known errors

        Args:
            scout_checkpoint: Path to scout checkpoint file
            guard_checkpoint: Path to guard checkpoint file
            checkpoint_prefix: Prefix for checkpoint files

        Returns:
            Tuple of (best_scout_genome, best_guard_genome)
        """
        import copy
        import os
        import random
        import pickle
        import time

        # Create directory structure
        os.makedirs(os.path.dirname("scout_" + checkpoint_prefix), exist_ok=True)
        os.makedirs(os.path.dirname("guard_" + checkpoint_prefix), exist_ok=True)

        # Load scout population
        if scout_checkpoint and os.path.exists(scout_checkpoint):
            print(f"Loading scout checkpoint from {scout_checkpoint}")
            scout_pop = neat.Checkpointer.restore_checkpoint(scout_checkpoint)
        else:
            print("Creating new scout population")
            scout_pop = neat.Population(self.config)

        # Add reporters to scout population
        scout_pop.add_reporter(neat.StdOutReporter(True))
        scout_stats = neat.StatisticsReporter()
        scout_pop.add_reporter(scout_stats)
        scout_pop.add_reporter(neat.Checkpointer(5, filename_prefix="scout_" + checkpoint_prefix))

        # ========== ULTIMATE GUARD HANDLING ==========
        # Try to load the best guard genome from file first
        if os.path.exists("best_guard_genome.pkl"):
            print("✅ Loading best guard genome from file")
            try:
                with open('best_guard_genome.pkl', 'rb') as f:
                    best_guard_genome = pickle.load(f)
                    self.best_guard_genome = best_guard_genome
                    self.best_guard_fitness = getattr(best_guard_genome, 'fitness', 0)
                    print(f"✅ Loaded best guard genome with fitness {self.best_guard_fitness}")
            except Exception as e:
                print(f"⚠️ Error loading best guard genome file: {str(e)}")

        # If we don't have a best guard yet, try to extract it from checkpoint
        if self.best_guard_genome is None and guard_checkpoint and os.path.exists(guard_checkpoint):
            print(f"⚠️ Attempting to extract best genome from {guard_checkpoint}")
            try:
                # Create a temporary population to extract the best genome
                temp_pop = neat.Checkpointer.restore_checkpoint(guard_checkpoint)
                if temp_pop.population:
                    # Extract the best genome
                    best_guard = max(temp_pop.population.values(), 
                                     key=lambda x: getattr(x, 'fitness', -float('inf')))
                    if hasattr(best_guard, 'fitness') and best_guard.fitness is not None:
                        self.best_guard_genome = best_guard
                        self.best_guard_fitness = best_guard.fitness
                        print(f"✅ Recovered best guard genome with fitness {self.best_guard_fitness}")
                        # Save it for future use
                        with open('best_guard_genome.pkl', 'wb') as f:
                            pickle.dump(self.best_guard_genome, f)
            except Exception as e:
                print(f"⚠️ Error extracting from guard checkpoint: {str(e)}")

        # Create a completely fresh guard population
        print("🔄 Creating fresh guard population from scratch")
        guard_pop = neat.Population(self.config)

        # Create new reporters for guard population - AVOID USING OLD REPORTERS
        guard_pop.add_reporter(neat.StdOutReporter(True))
        guard_stats = neat.StatisticsReporter()
        guard_pop.add_reporter(guard_stats)
        guard_pop.add_reporter(neat.Checkpointer(5, filename_prefix="guard_" + checkpoint_prefix))

        # Inject the best genome if we have one
        if self.best_guard_genome is not None:
            print("✅ Injecting best guard genome into new population")
            first_id = next(iter(guard_pop.population))
            guard_pop.population[first_id] = copy.deepcopy(self.best_guard_genome)

        # Early stopping variables
        stagnation_threshold = 10
        last_improvement_gen = 0
        best_combined_fitness = self.best_scout_fitness + self.best_guard_fitness

        # Training loop
        for generation in range(self.num_generations):
            print(f"\n======== Generation {generation} ========")

            # Create dummy fitness functions
            def scout_fitness_function(genomes, config):
                pass

            def guard_fitness_function(genomes, config):
                pass

            # Evaluate scout genomes against best guard
            if self.best_guard_genome:
                best_guard = self.best_guard_genome
            else:
                best_guard = list(guard_pop.population.values())[0]

            print(f"Evaluating {len(scout_pop.population)} scout genomes...")
            for genome_id, scout_genome in scout_pop.population.items():
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                scout_fitness = 0

                for trial in range(self.num_trials):
                    trial_seed = seed_base + (trial * 1000)
                    sf, _ = self._evaluate_single_competitive_trial(
                        scout_genome, best_guard, self.config, trial_seed)
                    scout_fitness += sf

                scout_genome.fitness = scout_fitness / self.num_trials

            # Update best scout
            current_best_scout = max(scout_pop.population.values(), key=lambda x: x.fitness)
            if current_best_scout.fitness > self.best_scout_fitness:
                print(f"📈 New best scout: {current_best_scout.fitness:.2f} (previous: {self.best_scout_fitness:.2f})")
                self.best_scout_fitness = current_best_scout.fitness
                self.best_scout_genome = current_best_scout
                with open('best_scout_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_scout_genome, f)

            # Evaluate guard genomes against best scout
            if self.best_scout_genome:
                best_scout = self.best_scout_genome
            else:
                best_scout = list(scout_pop.population.values())[0]

            print(f"Evaluating {len(guard_pop.population)} guard genomes...")
            for genome_id, guard_genome in guard_pop.population.items():
                seed_base = abs(hash(str(genome_id) + str(generation))) % 10000
                guard_fitness = 0

                for trial in range(self.num_trials):
                    trial_seed = seed_base + (trial * 1000)
                    _, gf = self._evaluate_single_competitive_trial(
                        best_scout, guard_genome, self.config, trial_seed)
                    guard_fitness += gf

                guard_genome.fitness = guard_fitness / self.num_trials

            # Update best guard
            current_best_guard = max(guard_pop.population.values(), key=lambda x: x.fitness)
            if current_best_guard.fitness > self.best_guard_fitness:
                print(f"📈 New best guard: {current_best_guard.fitness:.2f} (previous: {self.best_guard_fitness:.2f})")
                self.best_guard_fitness = current_best_guard.fitness
                self.best_guard_genome = current_best_guard
                with open('best_guard_genome.pkl', 'wb') as f:
                    pickle.dump(self.best_guard_genome, f)

            # Record fitness history
            self.fitness_history['scout'].append(self.best_scout_fitness)
            self.fitness_history['guard'].append(self.best_guard_fitness)

            # Check for improvement
            current_combined_fitness = self.best_scout_fitness + self.best_guard_fitness
            if current_combined_fitness > best_combined_fitness:
                print(f"📊 Fitness improvement: {best_combined_fitness:.2f} -> {current_combined_fitness:.2f}")
                best_combined_fitness = current_combined_fitness
                last_improvement_gen = generation

            # Early stopping check
            if generation - last_improvement_gen >= stagnation_threshold:
                print(f"🛑 Stopping early at generation {generation} - no improvement for {stagnation_threshold} generations")
                break

            # Create next generations
            print("Evolving scout population...")
            scout_pop.run(scout_fitness_function, 1)

            # ========== ULTIMATE GUARD EVOLUTION ==========
            print("Evolving guard population...")
            try:
                # Try normal evolution
                guard_pop.run(guard_fitness_function, 1)
            except Exception as e:
                print(f"⚠️ Error evolving guard population: {str(e)}")
                print("🔄 Creating fresh guard population due to error...")

                # Save the best genome if it exists
                if self.best_guard_genome:
                    best_guard = self.best_guard_genome
                    best_fitness = self.best_guard_fitness
                else:
                    # Try to extract best genome from current population
                    try:
                        best_guard = max(guard_pop.population.values(), 
                                         key=lambda x: getattr(x, 'fitness', -float('inf')))
                        best_fitness = getattr(best_guard, 'fitness', 0)
                    except:
                        # If that fails, just use any genome
                        best_guard = list(guard_pop.population.values())[0]
                        best_fitness = 0

                # Create completely new population WITHOUT reusing reporters
                guard_pop = neat.Population(self.config)

                # Add fresh reporters
                guard_pop.add_reporter(neat.StdOutReporter(True))
                guard_pop.add_reporter(neat.StatisticsReporter())
                guard_pop.add_reporter(neat.Checkpointer(5, filename_prefix="guard_" + checkpoint_prefix))

                # Insert best genome
                first_id = next(iter(guard_pop.population))
                guard_pop.population[first_id] = copy.deepcopy(best_guard)
                print(f"✅ Inserted best genome with fitness {best_fitness}")

                # Make sure all genomes have a fitness value to avoid None comparisons
                for genome_id, genome in guard_pop.population.items():
                    if genome_id != first_id and (not hasattr(genome, 'fitness') or genome.fitness is None):
                        genome.fitness = 0.0

        print("\n======== Training Complete ========")
        return self.best_scout_genome, self.best_guard_genome
    
    
    # NEW: Add helper method for single competitive trial evaluation
    def _evaluate_single_competitive_trial(self, scout_genome, guard_genome, config, seed):
        """
        Evaluate a single competitive trial between scout and guard

        Args:
            scout_genome: Scout genome
            guard_genome: Guard genome
            config: NEAT configuration
            seed: Random seed for the environment

        Returns:
            Tuple of (scout_fitness, guard_fitness)
        """
        scout_net = neat.nn.FeedForwardNetwork.create(scout_genome, config)
        guard_net = neat.nn.FeedForwardNetwork.create(guard_genome, config)

        scout_reward = 0
        guard_reward = 0

        env = gridworld.env(
            env_wrappers=[],
            render_mode=None,
            debug=True,
            novice=False
        )
        env.reset(seed=seed)

        # Get all initial agent IDs
        all_agents = list(env.agents)
        scout_id = all_agents[0]  # Assuming scout is first agent
        guard_ids = all_agents[1:]  # Assuming other agents are guards

        done = False
        step_counter = 0

        # Track if scout was captured (for guard reward)
        scout_captured = False

        while not done and step_counter < self.max_steps:
            step_counter += 1

            # Check if there are any agents left
            if len(env.agents) == 0:
                done = True
                break

            # Check if scout was captured
            if scout_id not in env.agents and not scout_captured:
                scout_captured = True
                scout_reward -= 50.0  # Penalty for getting caught
                guard_reward += 50.0  # Bonus for capturing
                done = True
                break

            # Get current agent - this might be None if all agents are done
            if not env.agents:
                done = True
                break

            current_agent = env.agent_selection

            try:
                # IMPORTANT: Only try to get observation if agent exists
                if current_agent in env.agents:
                    observation, reward, termination, truncation, info = env.last()

                    # Check if current agent is done
                    if termination or truncation:
                        # Agent is done, pass None as the action
                        env.step(None)
                        continue

                    # Process observation for NEAT
                    inputs = process_observation(observation)

                    if current_agent == scout_id:  # Scout agent
                        action = get_action_from_network(scout_net, inputs)
                        scout_reward += reward
                    else:  # Guard agents
                        action = get_action_from_network(guard_net, inputs)
                        guard_reward += reward

                    env.step(action)
                else:
                    # If current agent selection isn't in agents list, step with None
                    env.step(None)

            except KeyError as e:
                # This handles cases where an agent was just removed from the environment
                # Print debug info to help diagnose the issue
                print(f"KeyError encountered: {e}")
                print(f"Current agent: {current_agent}")
                print(f"Available agents: {env.agents}")
                print(f"Scout captured: {scout_captured}")

                # If scout is removed and we haven't registered it yet
                if scout_id not in env.agents and not scout_captured:
                    scout_captured = True
                    scout_reward -= 50.0
                    guard_reward += 50.0

                # Try to safely step the environment
                try:
                    env.step(None)
                except:
                    # If even that fails, just end the episode
                    done = True
                    break

        # Add survival bonus if scout survived
        if not scout_captured and step_counter >= self.max_steps:
            scout_reward += 50.0  # Bonus for surviving
            guard_reward -= 20.0  # Penalty for not capturing

        return scout_reward, guard_reward