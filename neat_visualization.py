import os
import matplotlib.pyplot as plt
import numpy as np
import neat
import graphviz

def plot_fitness_history(fitness_history, title="Fitness History", filename="fitness_history.png"):
    """
    Plot fitness history
    
    Args:
        fitness_history: Dictionary with 'scout' and 'guard' lists of fitness values
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history['scout'], label='Scout')
    plt.plot(fitness_history['guard'], label='Guard')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_species(stats, title="Speciation", filename="speciation.png"):
    """
    Visualize speciation
    
    Args:
        stats: NEAT statistics reporter
        title: Plot title
        filename: Output filename
    """
    if not stats.get_species_sizes():
        return
    
    plt.figure(figsize=(10, 6))
    species_sizes = stats.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T
    
    for i, curve in enumerate(curves):
        plt.plot(range(num_generations), curve, label=f"Species {i+1}")
    
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Size (Population)")
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig(filename)
    plt.close()

def plot_network(genome, config, filename="network"):
    """
    Visualize the network
    
    Args:
        genome: NEAT genome
        config: NEAT configuration
        filename: Output filename (without extension)
    """
    dot = neat.visualize.draw_net(config, genome, view=False)
    dot.format = 'png'
    dot.render(filename)

def visualize_agent_behavior(genome, config, env, num_episodes=3, max_steps=500, 
                             is_scout=True, filename_prefix="agent_behavior"):
    """
    Visualize agent behavior in the environment
    
    Args:
        genome: NEAT genome
        config: NEAT configuration
        env: Gridworld environment
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
        is_scout: Whether the agent is a scout
        filename_prefix: Prefix for output files
    """
    from feature_extraction import process_observation, get_action_from_network
    import neat
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename_prefix) if os.path.dirname(filename_prefix) else ".", exist_ok=True)
    
    for episode in range(num_episodes):
        env.reset(seed=episode)
        
        # Get all agent IDs
        all_agents = list(env.agents)
        scout_id = all_agents[0]  # Assuming scout is first agent
        guard_ids = all_agents[1:]  # Assuming other agents are guards
        
        # Simple policies for non-visualized agents
        scout_policy = lambda obs: np.random.randint(0, 5)
        guard_policy = lambda obs: np.random.randint(0, 5)
        
        positions = {agent_id: [] for agent_id in all_agents}
        actions = {agent_id: [] for agent_id in all_agents}
        
        done = False
        step_counter = 0
        
        while not done and step_counter < max_steps:
            step_counter += 1
            
            # Get current agent
            current_agent = env.agent_selection
            
            # Get observation and info
            observation, reward, termination, truncation, info = env.last()
            
            # Record position
            if current_agent in positions:
                positions[current_agent].append(observation['location'])
            
            # Check if current agent is done
            if termination or truncation:
                # Agent is done, pass None as the action
                env.step(None)
                continue
            
            # Process observation for NEAT
            inputs = process_observation(observation)
            
            if is_scout:  # Visualizing scout
                if current_agent == scout_id:
                    action = get_action_from_network(net, inputs)
                else:
                    action = guard_policy(observation)
            else:  # Visualizing guard
                if current_agent == scout_id:
                    action = scout_policy(observation)
                else:
                    action = get_action_from_network(net, inputs)
            
            if current_agent in actions:
                actions[current_agent].append(action)
                
            env.step(action)
            
            # Check if all agents are done
            if env.agents == []:
                done = True
        
        # Plot positions
        plt.figure(figsize=(10, 10))
        
        # Plot scout path
        if scout_id in positions and len(positions[scout_id]) > 0:
            scout_positions = np.array(positions[scout_id])
            plt.plot(scout_positions[:, 0], scout_positions[:, 1], 'b-', label='Scout Path')
            plt.plot(scout_positions[0, 0], scout_positions[0, 1], 'bo', markersize=10, label='Scout Start')
            plt.plot(scout_positions[-1, 0], scout_positions[-1, 1], 'bx', markersize=10, label='Scout End')
        
        # Plot guard paths
        for i, guard_id in enumerate(guard_ids):
            if guard_id in positions and len(positions[guard_id]) > 0:
                guard_positions = np.array(positions[guard_id])
                plt.plot(guard_positions[:, 0], guard_positions[:, 1], f'C{i+1}-', label=f'Guard {i+1} Path')
                plt.plot(guard_positions[0, 0], guard_positions[0, 1], f'C{i+1}o', markersize=10, label=f'Guard {i+1} Start')
                plt.plot(guard_positions[-1, 0], guard_positions[-1, 1], f'C{i+1}x', markersize=10, label=f'Guard {i+1} End')
        
        plt.title(f"Episode {episode+1} - {'Scout' if is_scout else 'Guard'} Behavior")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{filename_prefix}_episode_{episode+1}.png")
        plt.close()