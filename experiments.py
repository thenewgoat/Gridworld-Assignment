import pandas as pd
import os
import matplotlib.pyplot as plt
from environment import generate_random_environment, plot_environment
import value_iteration
import policy_iteration
from tqdm import tqdm
import numpy as np


# --------------------------
# Function to run experiments over different environments
# --------------------------
def run_experiments(grid_sizes, complexities, num_runs=5, base_seed=42,
                    gamma=0.99, tolerance=1e-5, max_iter_vi=5000, max_iter_pi=500):
    """
    For each combination of grid_size and complexity, run both Value Iteration and Policy Iteration
    on the generated environment, save figures using standardized filenames, and record convergence data.
    """
    results = []
    
    # Create directory for saving environment images if it does not exist.
    if not os.path.exists("env_images"):
        os.makedirs("env_images")

    total_experiments = len(grid_sizes) * (len(complexities) - 1) * num_runs + len(grid_sizes)
    pbar = tqdm(total=total_experiments, desc="Total Experiments")
    
    for grid_size in grid_sizes:
        for complexity in complexities:

            if complexity == 0.0:
                actual_num_runs = 1
            else:
                actual_num_runs = num_runs
            for run in range(actual_num_runs):
                current_seed = base_seed + run
                # Generate a random environment
                env = generate_random_environment(grid_size, complexity, seed=current_seed)
                env_label = f"env_grid{grid_size}_complex{complexity}_run{run}"
                env_image_path = os.path.join("env_images", f"{env_label}.png")
                plot_environment(env, save_path=env_image_path)
                
                # --- Value Iteration ---
                V_val, iteration_values_val, avg_utilities_val = value_iteration.value_iteration(
                    (grid_size, grid_size),
                    env["rewards"],
                    env["walls"],
                    env["action_vectors"],
                    env["action_probabilities"],
                    gamma=gamma,
                    tolerance=tolerance,
                    max_iter=max_iter_vi
                )
                iterations_val = iteration_values_val[-1][0] if iteration_values_val else None
                final_avg_util_val = avg_utilities_val[-1] if avg_utilities_val else None
                policy_val = value_iteration.extract_policy(V_val, env["rewards"], env["walls"], (grid_size, grid_size), gamma=gamma)
                
                results.append({
                    "grid_size": grid_size,
                    "complexity": complexity,
                    "run": run,
                    "algorithm": "value_iteration",
                    "iterations": iterations_val,
                    "final_avg_utility": final_avg_util_val,
                    "env_label": env_label
                })
                
                # Generate filenames for Value Iteration plots
                vi_fig_dir = os.path.join("figures", "value_iteration")
                value_grid_filename = get_figure_filename(vi_fig_dir, "value_grid", grid_size, complexity, run)
                vi_conv_filename = get_figure_filename(vi_fig_dir, "convergence", grid_size, complexity, run)

                util_filename_vi = os.path.join("results", f"final_util_{env_label}_value_iteration.csv")
                policy_filename_vi = os.path.join("results", f"final_policy_{env_label}_value_iteration.csv")
                np.savetxt(util_filename_vi, V_val, delimiter=",", fmt="%.4f")
                np.savetxt(policy_filename_vi, policy_val, delimiter=",", fmt="%s")
                
                # Save the Value Iteration utility grid and convergence plots
                value_iteration.plot_value_grid(V_val, policy_val, env["walls"],
                                title="Value Iteration Utilities and Policy",
                                filename=value_grid_filename)
                value_iteration.plot_utility_convergence(avg_utilities_val, filename=vi_conv_filename)
                
                # --- Policy Iteration ---
                policy_pol, V_pol, iteration_values_pol, avg_utilities_pol = policy_iteration.policy_iteration(
                    (grid_size, grid_size),
                    env["rewards"],
                    env["walls"],
                    env["action_vectors"],
                    env["action_probabilities"],
                    gamma=gamma,
                    tolerance=tolerance,
                    max_iter=max_iter_pi
                )
                iterations_pol = iteration_values_pol[-1][0] if iteration_values_pol else None
                final_avg_util_pol = avg_utilities_pol[-1] if avg_utilities_pol else None
                
                results.append({
                    "grid_size": grid_size,
                    "complexity": complexity,
                    "run": run,
                    "algorithm": "policy_iteration",
                    "iterations": iterations_pol,
                    "final_avg_utility": final_avg_util_pol,
                    "env_label": env_label
                })
                
                # Generate filenames for Policy Iteration plots
                pi_fig_dir = os.path.join("figures", "policy_iteration")
                policy_grid_filename = get_figure_filename(pi_fig_dir, "policy_grid", grid_size, complexity, run)
                pi_conv_filename = get_figure_filename(pi_fig_dir, "convergence", grid_size, complexity, run)
                
                # Save the Policy Iteration policy grid and convergence plots
                policy_iteration.plot_policy_grid(policy_pol, V_pol, env["walls"],
                                 title="Policy Iteration Policy",
                                 filename=policy_grid_filename)
                policy_iteration.plot_utility_convergence(avg_utilities_pol, filename=pi_conv_filename)

                util_filename_pi = os.path.join("results", f"final_util_{env_label}_policy_iteration.csv")
                policy_filename_pi = os.path.join("results", f"final_policy_{env_label}_policy_iteration.csv")
                np.savetxt(util_filename_pi, V_pol, delimiter=",", fmt="%.4f")
                np.savetxt(policy_filename_pi, policy_pol, delimiter=",", fmt="%s")

                pbar.update(1)
    pbar.close()
    return results

def plot_experiment_results(results):
    """
    Aggregate experiment results and produce multiple plots:
    
    1. For each algorithm (value_iteration and policy_iteration), produce a figure with two subplots:
       a) Average convergence iterations vs. grid size (with separate curves for each complexity).
       b) Final average utility vs. grid size (with separate curves for each complexity).
       
    2. For each algorithm, produce heatmaps:
       a) Heatmap of average convergence iterations (rows: grid size, columns: complexity).
       b) Heatmap of final average utility (rows: grid size, columns: complexity).
       
    3. Produce an aggregated plot comparing both algorithms for convergence iterations.
    
    This function produces two sets of plots:
      - One including complexity = 0.0 ("incl")
      - One excluding complexity = 0.0 ("excl")
      
    All figures are saved in the "results" directory.
    """
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    df = pd.DataFrame(results)
    # Group results to compute means for iterations and final utility
    grouped = df.groupby(["grid_size", "complexity", "algorithm"]).agg({
        "iterations": "mean",
        "final_avg_utility": "mean"
    }).reset_index()
    
    # Two sets: one including complexity = 0.0 and one excluding it
    set_labels = {
        "incl": "Including Complexity 0.0",
        "excl": "Excluding Complexity 0.0"
    }
    grouped_sets = {
        "incl": grouped,
        "excl": grouped[grouped["complexity"] != 0]
    }
    
    # For each set (incl/excl) and for each algorithm, produce line plots and heatmaps
    for set_key, group_data in grouped_sets.items():
        for algo in group_data["algorithm"].unique():
            algo_df = group_data[group_data["algorithm"] == algo]
            
            # 1. Line Plots for Each Algorithm: Two subplots (iterations and final utility)
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
            # (a) Plot average convergence iterations vs. grid size
            for comp in sorted(algo_df["complexity"].unique()):
                comp_data = algo_df[algo_df["complexity"] == comp]
                axs[0].plot(comp_data["grid_size"], comp_data["iterations"], marker='o', label=f"Complexity {comp}")
            axs[0].set_xlabel("Grid Size (n x n)")
            axs[0].set_ylabel("Avg. Convergence Iterations")
            axs[0].set_title(f"{algo.capitalize()} Convergence vs. Grid Size\n({set_labels[set_key]})")
            axs[0].legend()
            
            # (b) Plot final average utility vs. grid size
            for comp in sorted(algo_df["complexity"].unique()):
                comp_data = algo_df[algo_df["complexity"] == comp]
                axs[1].plot(comp_data["grid_size"], comp_data["final_avg_utility"], marker='o', label=f"Complexity {comp}")
            axs[1].set_xlabel("Grid Size (n x n)")
            axs[1].set_ylabel("Avg. Final Utility")
            axs[1].set_title(f"{algo.capitalize()} Final Utility vs. Grid Size\n({set_labels[set_key]})")
            axs[1].legend()
            
            plt.suptitle(f"{algo.capitalize()} Results ({set_labels[set_key]})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = os.path.join("results", f"lineplots_{algo}_{set_key}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            
            # 2. Heatmaps for Each Algorithm
            # (a) Heatmap for convergence iterations
            pivot_iter = algo_df.pivot(index="grid_size", columns="complexity", values="iterations")
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(pivot_iter, cmap='viridis', origin='lower', aspect='auto')
            ax.set_title(f"{algo.capitalize()} - Heatmap of Convergence Iterations\n({set_labels[set_key]})")
            ax.set_xlabel("Complexity")
            ax.set_ylabel("Grid Size")
            ax.set_xticks(np.arange(len(pivot_iter.columns)))
            ax.set_xticklabels(pivot_iter.columns)
            ax.set_yticks(np.arange(len(pivot_iter.index)))
            ax.set_yticklabels(pivot_iter.index)
            plt.colorbar(cax, ax=ax, label='Avg. Convergence Iterations')
            heatmap_filename = os.path.join("results", f"heatmap_iterations_{algo}_{set_key}.png")
            plt.savefig(heatmap_filename, bbox_inches='tight')
            plt.close(fig)
            
            # (b) Heatmap for final average utility
            pivot_util = algo_df.pivot(index="grid_size", columns="complexity", values="final_avg_utility")
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(pivot_util, cmap='magma', origin='lower', aspect='auto')
            ax.set_title(f"{algo.capitalize()} - Heatmap of Final Average Utility\n({set_labels[set_key]})")
            ax.set_xlabel("Complexity")
            ax.set_ylabel("Grid Size")
            ax.set_xticks(np.arange(len(pivot_util.columns)))
            ax.set_xticklabels(pivot_util.columns)
            ax.set_yticks(np.arange(len(pivot_util.index)))
            ax.set_yticklabels(pivot_util.index)
            plt.colorbar(cax, ax=ax, label='Avg. Final Utility')
            heatmap_util_filename = os.path.join("results", f"heatmap_utility_{algo}_{set_key}.png")
            plt.savefig(heatmap_util_filename, bbox_inches='tight')
            plt.close(fig)
        
        # 3. Aggregated Plot Comparing Both Algorithms for convergence iterations for this set:
        fig, ax = plt.subplots(figsize=(8, 6))
        for algo in group_data["algorithm"].unique():
            algo_df = group_data[group_data["algorithm"] == algo]
            for comp in sorted(algo_df["complexity"].unique()):
                comp_data = algo_df[algo_df["complexity"] == comp]
                ax.plot(comp_data["grid_size"], comp_data["iterations"], marker='o', linestyle='--',
                        label=f"{algo.capitalize()}, Complexity {comp}")
        ax.set_xlabel("Grid Size (n x n)")
        ax.set_ylabel("Avg. Convergence Iterations")
        ax.set_title(f"Comparison of Convergence Iterations across Algorithms\n({set_labels[set_key]})")
        ax.legend()
        filename = os.path.join("results", f"comparison_convergence_{set_key}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    
    print("Result plots saved for both algorithms in the 'results' directory.")





# Helper functions
def get_figure_filename(base_dir, figure_type, grid_size, complexity, run, suffix="png"):
    """
    Generate a standardized filename.
    - base_dir: the directory where the figure should be saved (e.g., "figures/value_iteration")
    - figure_type: a short descriptor (e.g., "value_grid" or "convergence")
    - grid_size: integer grid size (n for an n x n grid)
    - complexity: the complexity parameter (e.g., 0.1, 0.2, ...)
    - run: the run number (e.g., 0, 1, ...)
    - suffix: file extension (default "png")
    """
    filename = f"{figure_type}_grid{grid_size}_complex{complexity}_run{run}.{suffix}"
    # Ensure the directory exists:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return os.path.join(base_dir, filename)