import numpy as np

def extract_viewcone_features(viewcone):
    """
    Extract features from the viewcone observation.
    
    Args:
        viewcone: 7x5 grid of integers representing the agent's view
        
    Returns:
        List of normalized features extracted from viewcone
    """
    # Initialize empty feature list
    features = []
    
    # Process each tile in the viewcone
    for y in range(viewcone.shape[0]):
        for x in range(viewcone.shape[1]):
            tile_value = viewcone[y, x]
            
            # Extract tile type (empty, recon, mission)
            tile_type = tile_value & 0b11  # Last 2 bits
            
            # Extract entity presence
            scout_present = 1 if (tile_value & 0b100) else 0  # Bit 2
            guard_present = 1 if (tile_value & 0b1000) else 0  # Bit 3
            
            # Extract walls
            right_wall = 1 if (tile_value & 0b10000) else 0  # Bit 4
            bottom_wall = 1 if (tile_value & 0b100000) else 0  # Bit 5
            left_wall = 1 if (tile_value & 0b1000000) else 0  # Bit 6
            top_wall = 1 if (tile_value & 0b10000000) else 0  # Bit 7
            
            # Encode tile type as one-hot
            empty_tile = 1 if tile_type == 1 else 0
            recon_point = 1 if tile_type == 2 else 0
            mission_point = 1 if tile_type == 3 else 0
            
            # Add extracted features to list
            features.extend([empty_tile, recon_point, mission_point, 
                            scout_present, guard_present,
                            right_wall, bottom_wall, left_wall, top_wall])
    
    # Each tile contributes 9 features, so total size is 7x5x9 = 315
    return features

def process_observation(observation):
    """
    Process the full observation dictionary into input features for NEAT networks.
    
    Args:
        observation: Dictionary containing viewcone, direction, scout flag, location, step
        
    Returns:
        List of normalized features to feed into the neural network
    """
    features = []
    
    # Process viewcone (7x5 grid)
    viewcone_features = extract_viewcone_features(observation['viewcone'])
    features.extend(viewcone_features)
    
    # Process direction (one-hot encoding)
    direction = observation['direction']
    direction_features = [0, 0, 0, 0]  # One-hot for 4 directions
    direction_features[direction] = 1
    features.extend(direction_features)
    
    # Process agent type
    features.append(observation['scout'])  # 1 if scout, 0 if guard
    
    # Process location (normalized to [0,1] range)
    x, y = observation['location']
    features.append(x / 15.0)  # Normalize to [0,1] assuming 16x16 grid
    features.append(y / 15.0)
    
    # Process step count (normalized to [0,1] assuming max steps of 500)
    features.append(min(observation['step'] / 500.0, 1.0))
    
    # Add an additional feature to match the expected 324 inputs
    # This is a placeholder feature set to 0
    features.append(0.0)
    
    # Ensure we have exactly 324 features
    assert len(features) == 324, f"Expected 324 features, but got {len(features)}"
    
    return features

def calculate_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_action_from_network(network, inputs):
    """
    Get action from network output
    
    Args:
        network: NEAT network
        inputs: List of input features
        
    Returns:
        Integer action (0-4)
    """
    # Ensure inputs has the correct length
    if len(inputs) != 324:
        # If not correct length, pad or truncate
        if len(inputs) < 324:
            inputs = inputs + [0.0] * (324 - len(inputs))
        else:
            inputs = inputs[:324]
    
    # Activate the network with inputs
    output = network.activate(inputs)
    
    # Return the index of the maximum output value as the action
    return np.argmax(output)

def find_entities_in_viewcone(viewcone):
    """
    Find positions of entities in the viewcone
    
    Args:
        viewcone: 7x5 grid of integers
        
    Returns:
        Dictionary with positions of scouts, guards, recon points, and mission points
    """
    scouts = []
    guards = []
    recon_points = []
    mission_points = []
    
    for y in range(viewcone.shape[0]):
        for x in range(viewcone.shape[1]):
            tile_value = viewcone[y, x]
            tile_type = tile_value & 0b11
            
            # Check for entities
            if tile_value & 0b100:  # Scout present
                scouts.append((x, y))
            
            if tile_value & 0b1000:  # Guard present
                guards.append((x, y))
            
            # Check for points
            if tile_type == 2:  # Recon point
                recon_points.append((x, y))
            elif tile_type == 3:  # Mission point
                mission_points.append((x, y))
    
    return {
        'scouts': scouts,
        'guards': guards,
        'recon_points': recon_points,
        'mission_points': mission_points
    }