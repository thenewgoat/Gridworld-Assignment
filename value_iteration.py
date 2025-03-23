import numpy as np
import matplotlib.pyplot as plt

def value_iteration(grid_size, rewards, walls, action_vectors, action_probabilities, gamma=0.99, tolerance=1e-6, max_iter=5000):
    """
    Runs Value Iteration on the given environment.
    Returns:
        V: 2D array of state utilities
        iteration_values: list of (iteration, delta) for plotting convergence
    """

    actions = [key for key in action_vectors]  # Up, Down, Left, Right

    # Initialize utilities
    V = np.zeros(grid_size)
    iteration_values = []
    average_utilities = []     # mean of all utilities per iteration

    for iteration in range(max_iter):
        delta = 0.0
        new_V = np.copy(V)

        for r in range(grid_size[0]):
            for c in range(grid_size[1]):

                if (r, c) in walls:
                    continue  # No update for walls

                best_action_value = float('-inf')

                # Evaluate each possible action (U,D,L,R)
                for action in actions:
                    total = 0.0
                    # Probability distribution of next moves
                    for move_action, prob in action_probabilities[action].items():
                        # Compute next cell
                        new_r = r + action_vectors[move_action][0]
                        new_c = c + action_vectors[move_action][1]

                        # If out of bounds or wall, stay in place
                        if (new_r, new_c) in walls or not (0 <= new_r < grid_size[0]) or not (0 <= new_c < grid_size[1]):
                            new_r, new_c = r, c

                        # Bellman backup for immediate reward + discounted future
                        total += prob * (rewards[r, c] + gamma * V[new_r, new_c])

                    best_action_value = max(best_action_value, total)

                new_V[r, c] = best_action_value
                delta = max(delta, abs(new_V[r, c] - V[r, c]))

        V = new_V
        iteration_values.append((iteration, delta))
        area = grid_size[0] * grid_size[1]
        average_utilities.append(np.mean(V) * area/ (area - len(walls)))

        # Stop if changes are small
        if delta < tolerance:
            break

    return V, iteration_values, average_utilities

def extract_policy(V, rewards, walls, grid_size, gamma=0.99):
    """
    Extracts the policy from the final utilities.
    Returns:
        policy: 2D array with the best action (as a character) for each state.
    """
    actions = ['U', 'D', 'L', 'R']
    action_vectors = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    action_probabilities = {
        'U': {'U': 0.8, 'L': 0.1, 'R': 0.1},
        'D': {'D': 0.8, 'L': 0.1, 'R': 0.1},
        'L': {'L': 0.8, 'U': 0.1, 'D': 0.1},
        'R': {'R': 0.8, 'U': 0.1, 'D': 0.1}
    }
    
    policy = np.empty(grid_size, dtype=object)
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            if (r, c) in walls:
                policy[r, c] = 'W'
            else:
                best_action = None
                best_value = float('-inf')
                # For each action, compute its value based on the current utilities V
                for action in actions:
                    total = 0.0
                    for move_action, prob in action_probabilities[action].items():
                        new_r = r + action_vectors[move_action][0]
                        new_c = c + action_vectors[move_action][1]
                        if (new_r, new_c) in walls or not (0 <= new_r < grid_size[0]) or not (0 <= new_c < grid_size[1]):
                            new_r, new_c = r, c
                        total += prob * (rewards[r, c] + gamma * V[new_r, new_c])
                    if total > best_value:
                        best_value = total
                        best_action = action
                policy[r, c] = best_action
    return policy


def plot_value_grid(V, policy, walls, title='Value Iteration Utilities and Policy', filename="figures/value_iteration.png"):
    """
    Plots the utility values in each cell on a 2D grid and overlays the policy arrows.
    Dynamically adjusts font sizes and uses fixed fractional offsets so that the arrow 
    and numeric utility values remain centered in the top and bottom halves of each cell.
    
    - V: 2D NumPy array of state utilities.
    - policy: 2D array of best actions for each state.
    - walls: list of (r, c) cells that are walls.
    - title: figure title.
    - filename: file path to save the figure.
    """
    rows, cols = V.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot heatmap of utilities
    cax = ax.imshow(V, cmap='coolwarm', origin='upper')
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, label='Utility Value')
    
    # Map actions to arrows.
    arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'W': 'W'}
    
    # Dynamically compute font sizes based on grid size.
    arrow_fontsize = max(8, min(14, 100/rows))
    value_fontsize = max(4, min(10, 40/rows))
    
    # Use fixed fractional offsets relative to each 1x1 cell:
    arrow_offset = 0.20   # Arrow centered in the top half (0.25 units above cell center)
    value_offset = 0.25   # Value centered in the bottom half (0.25 units below cell center)
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls:
                ax.text(c, r, 'W', ha='center', va='center', fontsize=value_fontsize, color='black')
            else:
                action = policy[r, c]
                arrow_str = arrow_map.get(action, '?')
                # Place arrow in the top half (centered at r - 0.25)
                ax.text(c, r - arrow_offset, arrow_str,
                        ha='center', va='center', fontsize=arrow_fontsize, color='black')
                # Place utility value in the bottom half (centered at r + 0.25)
                ax.text(c, r + value_offset, f'{V[r, c]:.2f}',
                        ha='center', va='center', fontsize=value_fontsize, color='black')
    
    # Set ticks and grid lines
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.set_ylim([rows - 0.5, -0.5])  # Keep origin at top-left
    
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_utility_convergence(average_utilities, filename="figures/value_iteration_convergence.png"):
    """
    Plots the mean utility values as a function of iteration.
    Dynamically adjusts marker size based on the number of iterations.
    
    - average_utilities: List of utility values (one per iteration).
    - filename: file path to save the figure.
    """
    num_iter = len(average_utilities)
    marker_size = max(2, 10 - (num_iter / 100))  # reduce marker size for many iterations
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(average_utilities, marker='*', linestyle='-', linewidth=0, markersize=marker_size)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Utility')
    ax.set_title('Utility Estimates as a Function of Iterations for Value Iteration')
    ax.grid(True)
    
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)