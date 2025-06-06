import pickle
import neat
import numpy as np
from feature_extraction import process_observation, get_action_from_network

class NeatNetwork:
    """NEAT Network wrapper for Scout and Guard agents"""
    
    def __init__(self, genome=None, config=None):
        """
        Initialize NEAT Network
        
        Args:
            genome: NEAT genome
            config: NEAT configuration
        """
        self.genome = genome
        self.config = config
        self.net = None
        
        if genome is not None and config is not None:
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    def activate(self, inputs):
        """
        Activate the network with inputs
        
        Args:
            inputs: List of input features
            
        Returns:
            List of output values
        """
        if self.net is None:
            raise ValueError("Network not initialized")
        
        return self.net.activate(inputs)
    
    def save(self, filename):
        """
        Save the genome to a file
        
        Args:
            filename: Path to save the genome
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.genome, f)
    
    @classmethod
    def load(cls, filename, config):
        """
        Load a genome from a file
        
        Args:
            filename: Path to the genome file
            config: NEAT configuration
            
        Returns:
            NeatNetwork instance
        """
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        
        return cls(genome, config)


def create_network_from_genome(genome, config):
    """
    Create a neural network from a genome
    
    Args:
        genome: NEAT genome
        config: NEAT configuration
        
    Returns:
        NeatNetwork instance
    """
    return NeatNetwork(genome, config)


def load_genome(filename):
    """
    Load a genome from a file
    
    Args:
        filename: Path to the genome file
        
    Returns:
        NEAT genome
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)