import os
import pickle
import neat
import numpy as np
from feature_extraction import process_observation, get_action_from_network

def save_genome(genome, filename):
    """
    Save a genome to a file
    
    Args:
        genome: NEAT genome
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(genome, f)

def load_genome(filename):
    """
    Load a genome from a file
    
    Args:
        filename: Input filename
        
    Returns:
        NEAT genome
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_network_from_genome(genome, config):
    """
    Create a neural network from a genome
    
    Args:
        genome: NEAT genome
        config: NEAT configuration
        
    Returns:
        NEAT neural network
    """
    return neat.nn.FeedForwardNetwork.create(genome, config)

def evaluate_network(net, env, num_episodes=10, max_steps=500, is_scout=True):
    """
    Evaluate a neural network in the environment
    
    Args:
        net: NEAT neural network
        env: Gridworld environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        is_scout: Whether the agent is a scout
        
    Returns:
        Average reward
    """
    total_reward = 0
    
    for episode in range(num_episodes):
        env.reset(seed=episode)
        
        # Get all agent IDs
        all_agents = list(env.agents)
        scout_id = all_agents[0]  # Assuming scout is first agent
        guard_ids = all_agents[1:]  # Assuming other agents are guards
        
        # Simple policies for non-evaluated agents
        random_policy = lambda obs: np.random.randint(0, 5)
        
        episode_reward = 0
        done = False
        step_counter = 0
        
        while not done and step_counter < max_steps:
            step_counter += 1
            
            # Process all agents
            for agent_id in list(env.agents):  # Create a copy of agents list
                if agent_id not in env.agents:  # Skip if agent was removed
                    continue
                    
                observation, reward, termination, truncation, info = env.last()
                
                if is_scout and agent_id == scout_id:  # Evaluating scout
                    inputs = process_observation(observation)
                    action = get_action_from_network(net, inputs)
                    episode_reward += reward
                elif not is_scout and agent_id != scout_id:  # Evaluating guard
                    inputs = process_observation(observation)
                    action = get_action_from_network(net, inputs)
                    episode_reward += reward
                else:  # Other agents use random policy
                    action = random_policy(observation)
                
                env.step(action)
                
                # Check if environment is done
                if termination or truncation:
                    done = True
                    break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

def setup_directories():
    """
    Create necessary directories for outputs
    """
    directories = ['checkpoints', 'visualizations', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)