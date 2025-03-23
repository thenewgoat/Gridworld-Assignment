import numpy as np
import random
import matplotlib.pyplot as plt

# --------------------------
# Question 1: Default Environment
# --------------------------
def create_environment():
    grid_size = (6, 6) # 5x5 grid

    start_position = (3, 2)
    
    # Define rewards
    rewards = np.full(grid_size, -0.05)  # Default white square reward
    
    # Assign specific rewards
    reward_positions = {
        (0, 0): 1, (0, 2): 1, (0, 5): 1,
        (1, 1): -1, (1, 3): 1, (1, 5): -1, 
        (2, 2): -1, (2, 4): 1, 
        (3, 3): -1, (3, 5): 1, 
        (4, 4): 1
    }
    
    # Update rewards for specific positions
    for pos, value in reward_positions.items():
        rewards[pos] = value
    
    # Define walls (impassable states)
    walls = [(0, 1), (1, 4), (4, 1), (4, 2), (4, 3)]

    action_vectors = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    # Transition model for each intended action
    action_probabilities = {
        'U': {'U': 0.8, 'L': 0.1, 'R': 0.1},
        'D': {'D': 0.8, 'L': 0.1, 'R': 0.1},
        'L': {'L': 0.8, 'U': 0.1, 'D': 0.1},
        'R': {'R': 0.8, 'U': 0.1, 'D': 0.1}
    }
    
    return grid_size, rewards, walls, start_position, action_vectors, action_probabilities


# --------------------------
# Question 2: Custom Environments
# --------------------------


def create_environment_custom(grid_size, start_position, reward_positions, walls, default_reward=-0.05):
    """
    Create an environment with a square grid of given size.
    reward_positions: dict with keys as (i,j) and values as reward values.
    walls: list of (i,j) positions that are impassable.
    """
    rewards = np.full((grid_size, grid_size), default_reward)
    for pos, value in reward_positions.items():
        rewards[pos] = value

    action_vectors = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    action_probabilities = {
         'U': {'U': 0.8, 'L': 0.1, 'R': 0.1},
         'D': {'D': 0.8, 'L': 0.1, 'R': 0.1},
         'L': {'L': 0.8, 'U': 0.1, 'D': 0.1},
         'R': {'R': 0.8, 'U': 0.1, 'D': 0.1}
    }
    
    return {
        "grid_size": grid_size,
        "rewards": rewards,
        "walls": walls,
        "start_position": start_position,
        "action_vectors": action_vectors,
        "action_probabilities": action_probabilities,
        "reward_positions": reward_positions
    }



def generate_random_environment(grid_size, complexity, seed=42, default_reward=-0.05, feature_walls = 0.3333, feature_rewards = 0.3333):
    """
    Generates a random maze-like environment.
    
    Parameters:
      grid_size: int (grid will be grid_size x grid_size)
      complexity: float between 0 and 1 representing the fraction of cells that become features.
                  Features include walls, positive rewards, and negative rewards.
      seed: random seed for replicability.
      
    Returns:
      A dictionary representing the environment.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    total_cells = grid_size * grid_size
    if complexity == 1.0:
        total_cells -= 1
    num_features = round(complexity * total_cells)
    # Divide features: 1/3 walls, 1/3 reward, 1/3 penalty
    num_walls = round(num_features * feature_walls)
    num_positive = round(num_features * feature_rewards)
    num_negative = num_features - num_positive - num_walls

    # Define a start position (center of grid)
    start_position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
    
    # Create a list of available positions (exclude start_position)
    available_positions = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != start_position]
    random.shuffle(available_positions)
    
    # Assign walls
    walls = available_positions[:num_walls]
    available_positions = available_positions[num_walls:]
    
    # Assign rewards
    reward_positions = {}
    # Positive rewards (+1)
    for _ in range(num_positive):
        pos = available_positions.pop()
        reward_positions[pos] = 1
    # Negative rewards (-1)
    for _ in range(num_negative):
        pos = available_positions.pop()
        reward_positions[pos] = -1

    # Create the environment using the custom creator
    env = create_environment_custom(grid_size, start_position, reward_positions, walls, default_reward)
    return env

# --------------------------
# Plotting function for environment
# --------------------------
def plot_environment(env, save_path=None):
    """
    Visualize the grid environment.
    Walls are grey, positive rewards (1) are green, negative rewards (-1) are orange,
    and default rewards (-0.05) are white.
    """
    grid_size = env['grid_size']
    rewards = env['rewards']
    walls = env['walls']
    
    # Build a color grid based on reward values and walls
    color_grid = np.empty((grid_size, grid_size), dtype=object)
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in walls:
                color_grid[i, j] = 'grey'
            elif rewards[i, j] == 1:
                color_grid[i, j] = 'green'
            elif rewards[i, j] == -1:
                color_grid[i, j] = 'orange'
            else:
                color_grid[i, j] = 'white'
    
    # Plot each cell as a colored square
    fig, ax = plt.subplots()
    for i in range(grid_size):
        for j in range(grid_size):
            # Invert the y-axis to have (0,0) at the top-left
            rect = plt.Rectangle([j, grid_size - 1 - i], 1, 1, facecolor=color_grid[i, j], edgecolor='black')
            ax.add_patch(rect)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
