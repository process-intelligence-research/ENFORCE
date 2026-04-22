import matplotlib.pyplot as plt
import numpy as np
import os


def plot_all_results(
    losses_runs_dict, metrics_runs_dict,
    test_inputs_list_dict, test_outputs_list_dict,
    test_predictions_list_dict, test_prediction_before_projection_list_dict,
    output_dir, functions, train_inputs, train_outputs,
    test_inputs, test_outputs,
    save=True, left_limit=-2.0, right_limit=2.0, supervised=True,
):
    PLOT = True  # Ensure that plotting is enabled
    EPOCHS = len(losses_runs_dict[next(iter(losses_runs_dict))][0])
    epochs_range = np.arange(1, EPOCHS+1)

    modes = list(losses_runs_dict.keys())
    num_functions = len(functions)

    # Plot average losses with error bars including small horizontal delimiters
    if PLOT:
        # Plot Losses of both constrained and unconstrained together
        losses_plot = plt.figure(figsize=(12, 6))

        loss_data_plot = losses_plot.add_subplot(121)

        # Handle modes accordingly
        for mode in modes:
            if mode == 'constrained':
                # Constrained case: plot before and after projection
                losses_runs = losses_runs_dict[mode]
                # Convert losses to numpy arrays
                loss_data_array = np.array([
                    [
                        loss['loss_data_after_projection'] 
                        for loss in losses
                    ] 
                    for losses in losses_runs
                ])  # Shape (N, epochs)
                loss_displacement_array = np.array([
                    [
                        loss['loss_displacement'] 
                        for loss in losses
                        ] 
                        for losses in losses_runs
                ])  # Shape (N, epochs)
                loss_before_projection_array = np.array([
                    [
                        loss['loss_data_before_projection'] 
                        for loss in losses
                    ] 
                    for losses in losses_runs
                ])  # Shape (N, epochs)

                # Compute mean and std over runs
                mean_loss_data = np.mean(loss_data_array, axis=0)
                std_loss_data = np.std(loss_data_array, axis=0)

                mean_loss_displacement = np.mean(
                    loss_displacement_array, axis=0)
                std_loss_displacement = np.std(
                    loss_displacement_array, axis=0)

                mean_loss_before_projection = np.mean(
                    loss_before_projection_array, axis=0)
                std_loss_before_projection = np.std(
                    loss_before_projection_array, axis=0)

                # Plot after projection
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_data,
                    color='blue',
                    linestyle='-',
                    label='Constrained After Projection'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_data - std_loss_data,
                    mean_loss_data + std_loss_data,
                    color='blue',
                    alpha=0.1
                )
                # Plot before projection
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_before_projection,
                    color='blue',
                    linestyle='--',
                    label='Constrained Before Projection'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_before_projection - std_loss_before_projection,
                    mean_loss_before_projection + std_loss_before_projection,
                    color='blue',
                    alpha=0.1
                )
            elif mode == 'unconstrained':
                # Unconstrained case: only one loss
                losses_runs = losses_runs_dict[mode]
                loss_unconstrained_array = np.array([[loss['loss_unconstrained'] for loss in losses] for losses in losses_runs])  # Shape (N, epochs)
                mean_loss_unconstrained = np.mean(loss_unconstrained_array, axis=0)
                std_loss_unconstrained = np.std(loss_unconstrained_array, axis=0)
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_unconstrained,
                    color='green',
                    linestyle='-',
                    label='Unconstrained'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_unconstrained - std_loss_unconstrained,
                    mean_loss_unconstrained + std_loss_unconstrained,
                    color='green',
                    alpha=0.1
                )

        loss_data_plot.set_title("Loss Data" if supervised else "Objective Value")
        loss_data_plot.set_xlabel("Epoch")
        loss_data_plot.set_ylabel("Loss") if supervised else loss_data_plot.set_ylabel("Objective Value")
        loss_data_plot.legend()

        # Second Subplot: Projection Displacement with Shaded Standard Deviation
        proj_displacement = losses_plot.add_subplot(122)
        if 'constrained' in modes:
            mean_loss_displacement = np.mean(loss_displacement_array, axis=0)
            std_loss_displacement = np.std(loss_displacement_array, axis=0)
            proj_displacement.plot(
                epochs_range,
                mean_loss_displacement,
                color='red',
                linestyle='-',
                label='Constrained Projection Displacement'
            )
            proj_displacement.fill_between(
                epochs_range,
                mean_loss_displacement - std_loss_displacement,
                mean_loss_displacement + std_loss_displacement,
                color='red',
                alpha=0.1
            )
            proj_displacement.set_title("Projection Displacement")
            proj_displacement.set_xlabel("Epoch")
            proj_displacement.set_ylabel("Loss") 
            proj_displacement.legend()
        else:
            proj_displacement.axis('off')

        # Save the plot
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(output_dir, "plots", "losses_comparison.png"))
        # plt.close()

        ##
        # Plot Losses in log scale
        losses_plot_log = plt.figure(figsize=(12, 6))

        # First Subplot: Loss Data with Shaded Standard Deviation
        loss_data_plot = losses_plot_log.add_subplot(121)

        # Handle modes accordingly
        for mode in modes:
            if mode == 'constrained':
                if not supervised:
                    mean_loss_data = np.arcsinh(mean_loss_data)
                    mean_loss_before_projection = np.arcsinh(mean_loss_before_projection)
                    std_loss_data = np.arcsinh(std_loss_data)
                    std_loss_before_projection = np.arcsinh(std_loss_before_projection)
                # Constrained case: plot before and after projection
                # Plot after projection
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_data,
                    color='blue',
                    linestyle='-',
                    label='Constrained After Projection'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_data - std_loss_data,
                    mean_loss_data + std_loss_data,
                    color='blue',
                    alpha=0.1
                )
                # Plot before projection
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_before_projection,
                    color='blue',
                    linestyle='--',
                    label='Constrained Before Projection'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_before_projection - std_loss_before_projection,
                    mean_loss_before_projection + std_loss_before_projection,
                    color='blue',
                    alpha=0.1
                )
            elif mode == 'unconstrained':
                if not supervised:
                    mean_loss_unconstrained = np.arcsinh(mean_loss_unconstrained)
                    std_loss_unconstrained = np.arcsinh(std_loss_unconstrained)
                # Unconstrained case: only one loss
                loss_data_plot.plot(
                    epochs_range,
                    mean_loss_unconstrained,
                    color='green',
                    linestyle='-',
                    label='Unconstrained'
                )
                loss_data_plot.fill_between(
                    epochs_range,
                    mean_loss_unconstrained - std_loss_unconstrained,
                    mean_loss_unconstrained + std_loss_unconstrained,
                    color='green',
                    alpha=0.1
                )

        if supervised:
            loss_data_plot.set_yscale("log") # Set y-axis to log scale
        loss_data_plot.set_title("Loss Data" if supervised else "Objective Value")
        loss_data_plot.set_xlabel("Epoch")
        loss_data_plot.set_ylabel("Loss") if supervised else loss_data_plot.set_ylabel("arcsinh(Objective Value)")
        loss_data_plot.legend()

        # Second Subplot: Projection Displacement with Shaded Standard Deviation
        proj_displacement = losses_plot_log.add_subplot(122)
        if 'constrained' in modes:
            proj_displacement.plot(
                epochs_range,
                mean_loss_displacement,
                color='red',
                linestyle='-',
                label='Constrained Projection Displacement'
            )
            proj_displacement.fill_between(
                epochs_range,
                mean_loss_displacement - std_loss_displacement,
                mean_loss_displacement + std_loss_displacement,
                color='red',
                alpha=0.1
            )
            proj_displacement.set_yscale("log")  # Set y-axis to log scale
            proj_displacement.set_title("Projection Displacement")
            proj_displacement.set_xlabel("Epoch")
            proj_displacement.set_ylabel("Loss")
            proj_displacement.legend()
        else:
            proj_displacement.axis('off')

        # Save the plot
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(output_dir, "plots", "losses_log_comparison.png"))
        # plt.close()

        ##
        # Dataset distribution plot
        x1_grid = np.linspace(left_limit, right_limit, 1000)
        ncols = min(num_functions, 2)
        nrows = (num_functions + ncols - 1) // ncols

        dataset_distribution, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        if num_functions == 1:
            axes = [axes]  # Ensure axes is iterable
        else:
            axes = axes.flatten()

        # UNCOMMENT THIS BLOCK TO PLOT TRUE FUNCTIONS (ONLY FOR 1D REGRESSION PROBLEMS)
        if len(functions) == 2:
            try:
                for i, func in enumerate(functions):
                    ax = axes[i]
                    # ax.plot(x1_grid, func(x1_grid), label='True Function')
                    ax.scatter(test_inputs_list_dict[modes[0]][0], test_outputs_list_dict[modes[0]][0][:, i], c="red", label='Test')
                    ax.scatter(train_inputs, train_outputs[:, i], c="blue", label='Training')
                    ax.set_title(f'y{i+1}')
                    ax.legend()
            except Exception as e:    
                print(f"Error plotting functions: {e}")
                # Handle the error (e.g., skip plotting or log the error)
                pass

        dataset_distribution.suptitle("Training Points and Predictions (Data is the Same Across Runs)")
        if save:
            plt.savefig(os.path.join(output_dir, "plots", "dataset_distribution.png"))
        # plt.close()

        # Compute mean and std of predictions over runs for both constrained and unconstrained
        for mode in modes:
            test_predictions_array = np.array(test_predictions_list_dict[mode])  # Shape: [N, num_samples, num_functions]
            mean_predictions = np.mean(test_predictions_array, axis=0)
            std_predictions = np.std(test_predictions_array, axis=0)

            test_inputs = test_inputs_list_dict[mode][0]  # Same inputs across runs
            test_outputs = test_outputs_list_dict[mode][0]  # Same outputs across runs

            if mode == 'constrained':
                # For before projection, compute mean and std
                test_prediction_before_array = np.array(test_prediction_before_projection_list_dict[mode])
                mean_predictions_before = np.mean(test_prediction_before_array, axis=0)
                std_predictions_before = np.std(test_prediction_before_array, axis=0)
                # Number of subplots per function: 2
                ncols = 2
                nrows = num_functions
                fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
                if num_functions == 1:
                    axes = np.array([axes])  # Ensure axes is 2D array
            else:
                # Number of subplots per function: 1
                ncols = min(num_functions, 2)
                nrows = (num_functions + ncols - 1) // ncols
                fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
                if num_functions == 1:
                    axes = np.array([[axes]])  # Ensure axes is 2D array
                elif nrows == 1:
                    axes = np.expand_dims(axes, axis=0)

            axes = axes.reshape(-1, ncols)  # Ensure axes is 2D array

            # UNCOMMENT THIS BLOCK TO PLOT TRUE FUNCTIONS (ONLY FOR 1D REGRESSION PROBLEMS)
            if len(functions) == 2:
                try:
                    for i, func in enumerate(functions):
                        if mode == 'constrained':
                            # axes[i, 0] for prediction
                            # axes[i, 1] for before and after projection
                            ax_pred = axes[i, 0]
                            ax_pred.plot(x1_grid, func(x1_grid), label='True Function')
                            ax_pred.errorbar(test_inputs, mean_predictions[:, i], yerr=std_predictions[:, i], fmt='o', ecolor='gray', capsize=3, label=f"{mode.capitalize()} Prediction")
                            ax_pred.set_title(f"y{i+1} - Prediction")
                            ax_pred.legend()

                            ax_proj = axes[i, 1]
                            ax_proj.plot(x1_grid, func(x1_grid), label='True Function')
                            ax_proj.errorbar(test_inputs, mean_predictions_before[:, i], yerr=std_predictions_before[:, i], fmt='o', ecolor='orange', capsize=3, label=f"{mode.capitalize()} Prediction Before Projection")
                            ax_proj.errorbar(test_inputs, mean_predictions[:, i], yerr=std_predictions[:, i], fmt='o', ecolor='gray', capsize=3, label=f"{mode.capitalize()} Prediction After Projection")
                            ax_proj.set_title(f"y{i+1} - Before and After Projection")
                            ax_proj.legend()
                        else:
                            ax = axes.flatten()[i]
                            ax.plot(x1_grid, func(x1_grid), label='True Function')
                            ax.errorbar(test_inputs, mean_predictions[:, i], yerr=std_predictions[:, i], fmt='o', ecolor='gray', capsize=3, label=f"{mode.capitalize()} Prediction")
                            ax.set_title(f"y{i+1} - Prediction")
                            ax.legend()

                    fig2.suptitle(f"Test Prediction ({mode.capitalize()})")
                    plt.tight_layout()
                    if save:
                        plt.savefig(os.path.join(output_dir, "plots", f"{mode}_prediction_projection.png"))
                    # plt.close()

                    # Constraint plot (for two functions, generalizes to pairwise combinations if more than two)
                    if num_functions >= 2:
                        from itertools import combinations
                        combs = list(combinations(range(num_functions), 2))
                        n_combs = len(combs)
                        ncols = min(n_combs, 2)
                        nrows = (n_combs + ncols - 1) // ncols
                        constraint_plot, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
                        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
                        for idx, (i, j) in enumerate(combs):
                            ax = axes[idx]
                            ax.plot(functions[i](x1_grid), functions[j](x1_grid), label='Constraint Curve')
                            ax.errorbar(mean_predictions[:, i], mean_predictions[:, j], xerr=std_predictions[:, i], yerr=std_predictions[:, j], fmt='o', ecolor='gray', capsize=3, label=f"{mode.capitalize()} Prediction")
                            ax.set_xlabel(f"y{i+1}")
                            ax.set_ylabel(f"y{j+1}")
                            ax.legend()
                        constraint_plot.suptitle(f"Constraints ({mode.capitalize()})")
                        plt.tight_layout()
                        if save:
                            plt.savefig(os.path.join(output_dir, "plots", f"{mode}_constraint.png"))
                        # plt.close()
                    else:
                        # Skip constraint plot if only one function
                        pass
                except Exception as e:
                    print(f"Error plotting functions: {e}")
                    # Handle the error (e.g., skip plotting or log the error)
                    pass

            ## Parity plot
            ncols = min(num_functions, 2)
            nrows = (num_functions + ncols - 1) // ncols
            parity, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i in range(num_functions):
                ax = axes[i]
                ax.errorbar(test_outputs[:, i], mean_predictions[:, i], xerr=0, yerr=std_predictions[:, i], fmt='o', ecolor='gray', capsize=3)
                min_val = min(test_outputs[:, i].min(), mean_predictions[:, i].min())
                max_val = max(test_outputs[:, i].max(), mean_predictions[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax.set_xlabel('True')
                ax.set_ylabel('Prediction')
                ax.set_title(f'y{i+1}')

            parity.suptitle(f'Parity Plots ({mode.capitalize()})')
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(output_dir, "plots", f"{mode}_parity_plots.png"))
            # plt.close()

        # Plot comparison between constrained and unconstrained if both modes are available
        if len(modes) == 2:
            # Parity plot comparison
            ncols = min(num_functions, 2)
            nrows = (num_functions + ncols - 1) // ncols
            parity_comparison, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i in range(num_functions):
                ax = axes[i]
                for mode, marker in zip(['constrained', 'unconstrained'], ['o', 's']):
                    test_outputs = test_outputs_list_dict[mode][0]
                    test_predictions_array = np.array(test_predictions_list_dict[mode])  # Shape: [N, num_samples, num_functions]
                    mean_predictions = np.mean(test_predictions_array, axis=0)
                    ax.scatter(test_outputs[:, i], mean_predictions[:, i], label=f"{mode.capitalize()}", marker=marker)
                min_val = min(test_outputs[:, i].min(), mean_predictions[:, i].min())
                max_val = max(test_outputs[:, i].max(), mean_predictions[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax.set_xlabel('True')
                ax.set_ylabel('Prediction')
                ax.set_title(f'y{i+1}')
                ax.legend()

            parity_comparison.suptitle('Parity Plots Comparison')
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(output_dir, "plots", "parity_plots_comparison.png"))
            # plt.close()
        else:
            # If only one mode, no comparison plot
            pass

        # Show plots if needed
        # plt.show()


def plot_sin_ineq_results(
    train_inputs,
    train_outputs,
    test_inputs,
    test_outputs,
    test_predictions_list_dict,
    test_prediction_before_projection_list_dict,
    output_dir,
    save=True,
    sin_ineq_left=0.0,
    sin_ineq_right=9.4248,
):
    """Dedicated 2×3 visualisation for the sin_ineq case study.

    Panels
    ------
    [0,0] Feasible corridor + training data (violations highlighted) + predictions
    [0,1] Before vs after Newton projection (ENFORCE only)
    [0,2] Parity plot: true vs predicted
    [1,0] Upper constraint residual  g₁ = ŷ − (x²+0.5)   (should be ≤ 0)
    [1,1] Lower constraint residual  g₂ = −(x²+0.5) − ŷ  (should be ≤ 0)
    [1,2] Training data with corridor: shows which labels violate the constraints
    """
    modes   = list(test_predictions_list_dict.keys())
    colors  = {"constrained": "tab:blue", "unconstrained": "tab:orange", "soft": "tab:green"}
    labels_ = {"constrained": "ENFORCE",  "unconstrained": "MLP", "soft": "Soft ($\\lambda_C=1$)"}

    # ── Sort test data by x for line plots ──────────────────────────────────
    idx  = np.argsort(test_inputs[:, 0])
    x_s  = test_inputs[idx, 0]
    y_s  = test_outputs[idx, 0]

    # ── Dense grid for smooth corridor / true-function curves ───────────────
    x_d      = np.linspace(sin_ineq_left, sin_ineq_right, 500)
    amp_d    = 1.0 + x_d ** 2 / (3.0 * np.pi ** 2)   # quadratic envelope A(x) = 1 + x²/(3π²)
    y_true_d = amp_d * np.sin(x_d)
    upper_d  =  amp_d
    lower_d  = -amp_d

    # ── Training-label violation mask ────────────────────────────────────────
    x_tr    = train_inputs[:, 0]
    y_tr    = train_outputs[:, 0]
    amp_tr  = 1.0 + x_tr ** 2 / (3.0 * np.pi ** 2)
    viol_tr = (y_tr > amp_tr) | (y_tr < -amp_tr)

    # ── Per-mode prediction statistics (sorted by x) ────────────────────────
    def _stats(arr_list):
        arr = np.array(arr_list)  # [runs, N_TEST, 1]
        return (np.mean(arr[:, idx, 0], axis=0),
                np.std( arr[:, idx, 0], axis=0))

    pred_stats = {m: _stats(test_predictions_list_dict[m]) for m in modes}

    # Before-projection stats for constrained mode
    bp_mean = bp_std = None
    if "constrained" in test_prediction_before_projection_list_dict:
        bp_mean, bp_std = _stats(
            test_prediction_before_projection_list_dict["constrained"]
        )

    # ── Constraint residual helpers ──────────────────────────────────────────
    def g1(x, y): return  y - (1.0 + x ** 2 / (3.0 * np.pi ** 2))    # upper:  ≤ 0
    def g2(x, y): return -(1.0 + x ** 2 / (3.0 * np.pi ** 2)) - y    # lower:  ≤ 0

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # helper: draw corridor on any axis
    def _draw_corridor(ax, label=True):
        ax.fill_between(x_d, lower_d, upper_d,
                        color="gray", alpha=0.12,
                        label="Feasible corridor" if label else None)
        ax.plot(x_d, upper_d, "k--", lw=1.2,
                label="$\\pm(1+x^2/(3\\pi^2))$ envelope" if label else None)
        ax.plot(x_d, lower_d, "k--", lw=1.2)

    # ── [0,0]  Corridor + ground truth + predictions ──────────────────────────
    ax = axes[0, 0]
    _draw_corridor(ax)
    ax.plot(x_d, y_true_d, "k-", lw=2.0,
            label="True $y = (1+x^2/(3\\pi^2))\\sin(x)$")
    for m in modes:
        mn, sd = pred_stats[m]
        ax.plot(x_s, mn, lw=2.0, color=colors[m], label=labels_[m])
        ax.fill_between(x_s, mn - sd, mn + sd, alpha=0.18, color=colors[m])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Predictions vs feasible corridor")
    ax.legend(loc="lower right")

    # ── [0,1]  Training labels with violations ────────────────────────────────
    ax = axes[0, 1]
    _draw_corridor(ax, label=False)
    ax.plot(x_d, y_true_d, "k-", lw=1.5, alpha=0.6, label="True function")
    ax.scatter(x_tr[~viol_tr], y_tr[~viol_tr],
               s=8, alpha=0.30, color="tab:blue", label="Feasible labels")
    ax.scatter(x_tr[viol_tr],  y_tr[viol_tr],
               s=14, alpha=0.80, color="red",
               label=f"Violating labels ({100 * viol_tr.mean():.1f}%)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"Training data: {viol_tr.sum()}/{len(y_tr)} labels violate corridor"
    )
    ax.legend()

    # ── [1,0]  Upper constraint residual g₁ ──────────────────────────────────
    ax = axes[1, 0]
    ax.axhline(0, color="red", lw=1.5, ls="--", label="$g_1 = 0$ (boundary)")
    for m in modes:
        mn   = pred_stats[m][0]
        g1v  = g1(x_s, mn)
        nviol = int((g1v > 1e-4).sum())
        ax.plot(x_s, g1v, lw=1.6, color=colors[m],
                label=f"{labels_[m]} — violations: {nviol}/{len(x_s)}")
    g1_max = max(max(g1(x_s, pred_stats[m][0]).max() for m in modes), 0)
    if g1_max > 1e-4:
        ax.fill_between(x_s, 0, g1_max * 1.05,
                        alpha=0.07, color="red", label="Infeasible ($g_1 > 0$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g_1 = \\hat{y} - (1+x^2/(3\\pi^2))$")
    ax.set_title("Upper constraint residual $g_1 \\leq 0$")
    ax.legend()

    # ── [1,1]  Lower constraint residual g₂ ──────────────────────────────────
    ax = axes[1, 1]
    ax.axhline(0, color="red", lw=1.5, ls="--", label="$g_2 = 0$ (boundary)")
    for m in modes:
        mn   = pred_stats[m][0]
        g2v  = g2(x_s, mn)
        nviol = int((g2v > 1e-4).sum())
        ax.plot(x_s, g2v, lw=1.6, color=colors[m],
                label=f"{labels_[m]} — violations: {nviol}/{len(x_s)}")
    g2_max = max(max(g2(x_s, pred_stats[m][0]).max() for m in modes), 0)
    if g2_max > 1e-4:
        ax.fill_between(x_s, 0, g2_max * 1.05,
                        alpha=0.07, color="red", label="Infeasible ($g_2 > 0$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g_2 = -(1+x^2/(3\\pi^2)) - \\hat{y}$")
    ax.set_title("Lower constraint residual $g_2 \\leq 0$")
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(output_dir, "plots", "sin_ineq_results.png"),
            dpi=150,
            bbox_inches="tight",
        )
