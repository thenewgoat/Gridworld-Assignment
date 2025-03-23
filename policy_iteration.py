import numpy as np
import matplotlib.pyplot as plt

def policy_iteration(grid_size, rewards, walls, action_vectors, action_probabilities, gamma=0.99, tolerance=1e-6, max_iter=500):
    """
    Runs Policy Iteration on the environment, with a full policy evaluation
    that accounts for the 0.8/0.1/0.1 transition model.
    Returns:
        policy: 2D array of best actions (arrows)
        V: 2D array of state utilities
        iteration_values: list tracking iteration and max change in V
    """
    
    actions = [key for key in action_vectors]  # Up, Down, Left, Right

    # Initialize a random policy
    #np.random.seed(4003)
    #policy = np.random.choice(actions, size=grid_size)
    
    # set default policy to all right
    policy = np.full(grid_size, 'R')

    # Any walls get a 'W' label so we skip them
    for (wr, wc) in walls:
        policy[wr, wc] = 'W'

    # Initialize V
    V = np.zeros(grid_size)
    iteration_values = []      # (iteration, delta)
    average_utilities = []
    iteration_count = 0

    while True:
        iteration_count += 1

        # 1) POLICY EVALUATION: Iteratively evaluate V^pi until stable
        while True:
            delta = 0.0
            new_V = np.copy(V)

            for r in range(grid_size[0]):
                for c in range(grid_size[1]):
                    # Skip walls
                    if (r, c) in walls:
                        continue

                    a = policy[r, c]
                    if a == 'W':
                        continue

                    # Evaluate V under the chosen action a
                    value_sum = 0.0
                    for move_action, prob in action_probabilities[a].items():
                        nr = r + action_vectors[move_action][0]
                        nc = c + action_vectors[move_action][1]
                        # If out of bounds or wall, stay in place
                        if (nr, nc) in walls or not (0 <= nr < grid_size[0]) or not (0 <= nc < grid_size[1]):
                            nr, nc = r, c
                        value_sum += prob * (rewards[r, c] + gamma * V[nr, nc])

                    new_V[r, c] = value_sum
                    delta = max(delta, abs(new_V[r, c] - V[r, c]))

            V = new_V

            # Track convergence
            iteration_values.append((iteration_count, delta))
            area = grid_size[0] * grid_size[1]
            average_utilities.append(np.mean(V) * area/ (area - len(walls)))
            # If values have converged enough under this policy, break
            if delta < tolerance:
                break

        # 2) POLICY IMPROVEMENT
        policy_stable = True
        new_policy = np.copy(policy)

        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                if (r, c) in walls:
                    continue

                # Current action
                old_action = policy[r, c]

                # Find best action
                best_action_val = float('-inf')
                best_action = old_action

                for candidate_a in actions:
                    if candidate_a == 'W':
                        continue

                    value_sum = 0.0
                    for move_action, prob in action_probabilities[candidate_a].items():
                        nr = r + action_vectors[move_action][0]
                        nc = c + action_vectors[move_action][1]
                        if (nr, nc) in walls or not (0 <= nr < grid_size[0]) or not (0 <= nc < grid_size[1]):
                            nr, nc = r, c
                        value_sum += prob * (rewards[r, c] + gamma * V[nr, nc])

                    if value_sum > best_action_val:
                        best_action_val = value_sum
                        best_action = candidate_a

                new_policy[r, c] = best_action

                if best_action != old_action:
                    policy_stable = False

        policy = new_policy
        iteration_values.append((iteration_count, delta))
        area = grid_size[0] * grid_size[1]
        average_utilities.append(np.mean(V) * area/ (area - len(walls)))

        # If policy didn't change, we've converged
        if policy_stable:
            break

    return policy, V, iteration_values, average_utilities

def plot_policy_grid(policy, V, walls, title='Policy Iteration Policy', filename="figures/policy_iteration.png"):
    """
    Plots a heatmap of V and overlays the policy arrows and numeric utility values in each cell.
    Dynamically adjusts font sizes and uses fixed fractional offsets so that the arrow 
    and numeric utility values remain centered in the top and bottom halves of each cell.
    
    - policy: 2D array with strings 'U','D','L','R' or 'W'
    - V: 2D NumPy array for the utility values (background heatmap)
    - walls: list of (r, c) cells that are walls
    - title: figure title
    - filename: file path to save the figure
    """
    arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'W': 'W'}
    
    rows, cols = V.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display V as a background heatmap
    cax = ax.imshow(V, cmap='coolwarm', origin='upper')
    ax.set_title(title)
    plt.colorbar(cax, ax=ax, label='Utility Value')
    
    # Compute dynamic font sizes based on grid size.
    arrow_fontsize = max(8, min(14, 100/rows))
    value_fontsize = max(4, min(10, 40/rows))
    
    # Use fixed fractional offsets relative to each cell (cell size = 1)
    arrow_offset = 0.25  # arrow centered in the top half (0.25 units above cell center)
    value_offset = 0.25  # value centered in the bottom half (0.25 units below cell center)
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls:
                ax.text(c, r, 'W', ha='center', va='center', fontsize=value_fontsize, color='black')
            else:
                action = policy[r, c]
                arrow_str = arrow_map.get(action, '?')
                # Place the arrow in the top half (centered at r - arrow_offset)
                ax.text(c, r - arrow_offset, arrow_str,
                        ha='center', va='center', fontsize=arrow_fontsize, color='black')
                # Place the numeric utility in the bottom half (centered at r + value_offset)
                ax.text(c, r + value_offset, f'{V[r, c]:.2f}',
                        ha='center', va='center', fontsize=value_fontsize, color='black')
    
    # Set ticks and grid lines
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.set_ylim([rows - 0.5, -0.5])
    
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_utility_convergence(average_utilities, filename="figures/policy_iteration_convergence.png"):
    """
    Plots the mean utility values as a function of iteration for policy iteration.
    Dynamically adjusts the marker size based on the number of iterations.
    
    - average_utilities: List of utility values (one per iteration)
    - filename: file path to save the figure
    """
    num_iter = len(average_utilities)
    marker_size = max(2, 10 - (num_iter / 100))  # Adjust marker size for many iterations
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(average_utilities, marker='*', linestyle='-', linewidth=0, markersize=marker_size)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Utility')
    ax.set_title('Utility Estimates as a Function of Iterations for Policy Iteration')
    ax.grid(True)
    
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)