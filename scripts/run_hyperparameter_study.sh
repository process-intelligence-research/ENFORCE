#!/bin/bash

# Hyperparameter study: sweeps WEIGHT_LOSS_DISPLACEMENT and TRAINING_TOLERANCE
# for the function_fitting problem.
#
# Usage (from repo root):
#   bash scripts/run_hyperparameter_study.sh

weight_loss_displacement_values=(0 0.001 0.1 0.25 0.5 0.75 1.0 2.0)
training_tolerance_values=(0.00001 0.0001 0.001 0.01 0.1 1)

config_file="configs/config_benchmarking.py"

# Backup original config
cp "$config_file" "${config_file}.bak"

# Disable plotting and set output directory for the study
sed -i 's/^PLOT = True/PLOT = False/' "$config_file"
sed -i '/^if PROBLEM in ("function_fitting")/,/^elif/ s|^[[:space:]]*TRAINING_DIR = .*|    TRAINING_DIR = "training_output/lambda_epsilon_5_runs"|' "$config_file"

for weight_loss_displacement in "${weight_loss_displacement_values[@]}"; do
    for training_tolerance in "${training_tolerance_values[@]}"; do

        # Patch WEIGHT_LOSS_DISPLACEMENT and TRAINING_TOLERANCE in the
        # function_fitting / extraction_column block
        sed -i '/^if PROBLEM in ("function_fitting", "extraction_column")/,/^elif/ s/^[[:space:]]*WEIGHT_LOSS_DISPLACEMENT = .*/    WEIGHT_LOSS_DISPLACEMENT = '"$weight_loss_displacement"'/' "$config_file"
        sed -i '/^if PROBLEM in ("function_fitting", "extraction_column")/,/^elif/ s/^[[:space:]]*TRAINING_TOLERANCE = .*/    TRAINING_TOLERANCE = '"$training_tolerance"'/' "$config_file"

        echo "Running with WEIGHT_LOSS_DISPLACEMENT=$weight_loss_displacement TRAINING_TOLERANCE=$training_tolerance"
        python scripts/run_benchmark.py

    done
done

# Restore original config
mv "${config_file}.bak" "$config_file"
