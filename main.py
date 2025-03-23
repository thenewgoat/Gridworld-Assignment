from environment import create_environment
import value_iteration
import policy_iteration
from experiments import run_experiments, plot_experiment_results
import pandas as pd

# For running experiments
import matplotlib



# Parameters
gamma = 0.99
tolerance = 1e-5

# Part 1: Default Environment

grid_size, rewards, walls, start_position, action_vectors, action_probabilities = create_environment()

# ----- Value Iteration -----
V_val, iters_val, value_average_utilities = value_iteration.value_iteration(grid_size, 
                                                                            rewards, 
                                                                            walls, 
                                                                            action_vectors, 
                                                                            action_probabilities, 
                                                                            gamma, 
                                                                            tolerance
                                                                            )
print("\n=== Value Iteration Utilities ===")
policy = value_iteration.extract_policy(V_val, rewards, walls, grid_size)
value_iteration.plot_value_grid(V_val, policy, walls, title = 'Value Iteration: Utilities and Policy')
value_iteration.plot_utility_convergence(value_average_utilities)
# plot iters_val to see how quickly it converged



# ----- Policy Iteration -----
policy, V_pol, iters_pol, policy_average_utilities = policy_iteration.policy_iteration(grid_size,
                                                                                       rewards, 
                                                                                       walls, 
                                                                                       action_vectors, 
                                                                                       action_probabilities, 
                                                                                       gamma, 
                                                                                       tolerance
                                                                                       )
print("\n=== Policy Iteration: Final Policy ===")
policy_iteration.plot_policy_grid(policy, V_pol, walls, title='Policy Iteration: Utilities and Policy')
policy_iteration.plot_utility_convergence(policy_average_utilities)


# Part 2: Custom Environments
print("Part 1 done! View plots in the 'figures' directory.")
print("Press Enter to continue with Part 2 of the assignment.")
input()


matplotlib.use('Agg')


# Explore different grid sizes and complexities
grid_sizes = [4, 6, 8, 10, 12, 14, 16, 18]       # For example, 6x6, 8x8, 10x10 grids
complexities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Fraction of cells with features (walls + rewards)
num_runs = 10                 # Number of runs for each combination for replicability

# Run experiments using both value iteration and policy iteration
results = run_experiments(grid_sizes, complexities, num_runs=num_runs,
                            base_seed=42, gamma = 0.99, tolerance = 1e-5,
                            max_iter_vi=5000, max_iter_pi=500)

# Save experiment results to CSV for further analysis if desired.
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results_combined.csv", index=False)

# Plot and save summary graphs.
plot_experiment_results(results)

print("Experiments complete. Check the 'env_images' directory, the CSV file, and the result plots for outputs.")
