#! /bin/bash

########################################
# Parameter Sweep Configuration
# -- Define lists of parameters to sweep over
########################################
TEST_VERSIONS=("normalized")  # TestLoss  -linearly_normalized
SEEDS=(42)

WITH_LOGITS=("1")
LAYERS_LIST=("" "8" "16")
NUM_GRIDS_LIST=("4 6")
GRID_MIN_LIST=("-3 -1.7")
GRID_MAX_LIST=("2 1.25")
SCALE_LIST=("8 0.5")
MODES=('custom')
RESIDUALS=(0)
DYNAMICS=(0)
NO_NORMALIZES=(1)
DROPOUTS=(0.15)

EPOCHS_LIST=(1000)
PATIENCE_LIST=(50)
BATCH_SIZES=(16384)
LEARNING_RATES=(5e-2)
LR_FACTORS=(0.5)           # Learning rate reduction factor
LR_PATIENCE_LIST=(8)       # Scheduler patience
OPTIMIZERS=("AdamW")
WEIGHT_DECAYS=(1e-4)
MOMENTUMS=("0.9")

########################################
# Runtime Configuration
########################################
MAX_PARALLEL_JOBS=1  # Number of parallel experiments (set to 1 for sequential)
THIS_DIR=$(dirname $(realpath $0))
RESULTS_DIR="$THIS_DIR/train/sweep_results"

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0
max_experiments=-1
no_pbar=0
assume_yes=
assume_no=

# Job control for parallel execution
wait_for_jobs() {
    local max_jobs=$1
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

get_running_jobs() {
    jobs -r | wc -l
}

dry_run () {
    if [ $dryrun -lt 1 ]; then
        eval "$*";
    fi
}
print_verbose () {
    if [ $verbose -ge 1 ]; then
        echo "$*";
    fi;
}
print_exec () {
    print_verbose [EXEC] "$*";
    dry_run "$*";
}

log_experiment() {
    local exp_num=$1
    local total_exp=$2
    local config_hash=$3
    local status=$4
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    if [ -z $THIS_DIR ]; then
        local THIS_DIR="."
    fi
    echo "[$timestamp] Experiment $exp_num/$total_exp (Hash: $config_hash) - $status" | tee -a "$RESULTS_DIR/sweep_log.txt"
}

usage () {
    echo NAME
    echo "      run-kan-sweep.sh [-h] [-d] [-v] [-p] [--max-experiments N] [-j N] [--no-pbar] [-y|-N]" 
    echo
    echo DESCRIPTION
    echo "      Runs a parameter sweep training multiple KAN models with different configurations."
    echo
    echo Parameters
    echo "      -h, --help             Prints out help"
    echo "      -d, --dryrun           Dry run of the script (shows what would be executed)"
    echo "      -v, --verbose          Prints the to be executed commands"
    echo "      -p, --purge            Purges any existing output files before generating them"
    echo "      --max-experiments N    Limit the total number of experiments to N"
    echo "      -j, --jobs N           Number of parallel jobs (default: from MAX_PARALLEL_JOBS in script)"
    echo "      --no-pbar              Disable progress bars (useful for reducing log file size)"
    echo "      -y / -N                Assume yes / no"
    echo
    echo NOTES
    echo "      To run experiments in parallel, either set MAX_PARALLEL_JOBS in the script"
    echo "      or use the -j flag. For example: -j 4 to run 4 experiments simultaneously."
    echo "      Terminal output is saved to terminal_output.txt in each model's folder."
}

while [ "$#" -gt 0 ] ; do
    case "$1" in
        -h|--help) 
            usage
            exit 0 ;;
        -d|--dry-run) 
            dryrun=1
            verbose=1
            shift ;;
        -v|--verbose) 
            verbose=1
            shift ;;
        -p|--purge) 
            purge=1
            shift ;;
        -j|--jobs)
            MAX_PARALLEL_JOBS=$2
            shift
            shift ;;
        --max-experiments)
            max_experiments=$2
            shift 
            shift ;;
        --no-pbar)
            no_pbar=1
            shift ;;
        -y)
            assume_yes=1
            shift ;;
        -N)
            assume_no=1
            shift ;;
        -*|--*=)  # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            exit 1 ;;
        *)  # preserve positional arguments
            PARAMS="$PARAMS $1"
            shift ;;
    esac
done

if [[ assume_no -ge 1 && assume_yes -ge 1 ]]; then
    echo "Error: only one of '-y', '-N' can be set at a time."
    exit -1;
fi

# set positional arguments in their proper place
eval set -- "$PARAMS"

# Validate parallel jobs setting
if [ $MAX_PARALLEL_JOBS -lt 1 ]; then
    echo "Error: MAX_PARALLEL_JOBS must be at least 1"
    exit 1
fi

if [ $MAX_PARALLEL_JOBS -gt 1 ]; then
    echo "========================================="
    echo "WARNING: Parallel Execution Enabled"
    echo "========================================="
    echo "Running $MAX_PARALLEL_JOBS experiments in parallel."
    echo "Make sure you have sufficient GPU memory!"
    echo "Consider setting CUDA_VISIBLE_DEVICES for each experiment"
    echo "or reduce MAX_PARALLEL_JOBS if you encounter OOM errors."
    echo "========================================="
    echo ""
fi

print_exec cd $(dirname $THIS_DIR)

# Create results directory
print_exec mkdir -p "$RESULTS_DIR"

# Calculate total number of experiments
total_combinations=1
for param_list in "${WITH_LOGITS[@]}" "${LAYERS_LIST[@]}" "${NUM_GRIDS_LIST[@]}" "${GRID_MIN_LIST[@]}" "${GRID_MAX_LIST[@]}" "${SCALE_LIST[@]}" "${MODES[@]}" "${RESIDUALS[@]}" "${DYNAMICS[@]}" "${NO_NORMALIZES[@]}" "${DROPOUTS[@]}" "${EPOCHS_LIST[@]}" "${PATIENCE_LIST[@]}" "${BATCH_SIZES[@]}" "${LEARNING_RATES[@]}" "${LR_FACTORS[@]}" "${LR_PATIENCE_LIST[@]}" "${OPTIMIZERS[@]}" "${WEIGHT_DECAYS[@]}" "${MOMENTUMS[@]}" "${TEST_VERSIONS[@]}" "${SEEDS[@]}"; do
    break
done

# Count total experiments
exp_count=0
for WITH_LOGIT in "${WITH_LOGITS[@]}"; do
for TEST_VERSION in "${TEST_VERSIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LAYERS in "${LAYERS_LIST[@]}"; do
            for NUM_GRIDS in "${NUM_GRIDS_LIST[@]}"; do
                for GRID_MIN in "${GRID_MIN_LIST[@]}"; do
                    for GRID_MAX in "${GRID_MAX_LIST[@]}"; do
                        for SCALE in "${SCALE_LIST[@]}"; do
                            for MODE in "${MODES[@]}"; do
                                for RESIDUAL in "${RESIDUALS[@]}"; do
                                    for DYNAMIC in "${DYNAMICS[@]}"; do
                                        for NO_NORMALIZE in "${NO_NORMALIZES[@]}"; do
                                            for DROPOUT in "${DROPOUTS[@]}"; do
                                                for EPOCHS in "${EPOCHS_LIST[@]}"; do
                                                    for PATIENCE in "${PATIENCE_LIST[@]}"; do
                                                        for BATCH in "${BATCH_SIZES[@]}"; do
                                                            for LR in "${LEARNING_RATES[@]}"; do
                                                                for LR_FACTOR in "${LR_FACTORS[@]}"; do
                                                                    for LR_PATIENCE in "${LR_PATIENCE_LIST[@]}"; do
                                                                        for OPTIMIZER in "${OPTIMIZERS[@]}"; do
                                                                            for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                                                                for MOMENTUM in "${MOMENTUMS[@]}"; do
                                                                                    ((exp_count++))
                                                                                    if [ $max_experiments -gt 0 ] && [ $exp_count -gt $max_experiments ]; then
                                                                                        break 20  # Break out of all loops
                                                                                    fi
                                                                                done
                                                                            done
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
done

total_experiments=$exp_count
if [ $max_experiments -gt 0 ] && [ $max_experiments -lt $total_experiments ]; then
    total_experiments=$max_experiments
fi

echo "========================================="
echo "KAN Parameter Sweep Configuration"
echo "========================================="
echo "Total experiments to run: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Results directory: $RESULTS_DIR"
echo "Dry run mode: $dryrun"
echo "========================================="

if [ $dryrun -eq 0 ]; then
    if   [[ -z $assume_no  && -z $assume_yes    ]]; then
        read -p "Do you want to proceed? (y/N): " -n 1 -r
        echo
    elif [[ -z $assume_no  && $assume_yes -ge 1 ]]; then
        read -p "Do you want to proceed? (y/N): " -n 1 -r <<< y
        echo
    elif [[ -z $assume_yes && $assume_no  -ge 1 ]]; then
        read -p "Do you want to proceed? (y/N): " -n 1 -r <<< N
        echo
    fi
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Initialize log file
if [ $dryrun -eq 0 ]; then
    echo "Parameter sweep started at $(date)" > "$RESULTS_DIR/sweep_log.txt"
    echo "Total experiments: $total_experiments" >> "$RESULTS_DIR/sweep_log.txt"
    echo "Max parallel jobs: $MAX_PARALLEL_JOBS" >> "$RESULTS_DIR/sweep_log.txt"
    echo "=========================================" >> "$RESULTS_DIR/sweep_log.txt"
fi

# Function to run a single experiment
run_experiment() {
    local exp_num=$1
    local total_exp=$2
    local exp_hash=$3
    local configs=$4
    local with_logit=$5
    local test_version=$6
    local use_no_pbar=$7
    
    # Determine output directory
    if [ -n "$test_version" ]; then
        test_dir_name="test_${test_version}"
    else
        test_dir_name="test_0"
    fi
    output_dir="$THIS_DIR/train/${exp_hash}/${test_dir_name}"
    terminal_output="${output_dir}/terminal_output.txt"
    
    echo ""
    echo "========================================="
    echo "[Job $$] Running Experiment $exp_num/$total_exp"
    echo "========================================="
    echo "Configuration Hash: $exp_hash"
    echo "Output Directory: $output_dir"
    echo "========================================="
    
    log_experiment $exp_num $total_exp $exp_hash "STARTED"
    
    # Run training pipeline
    local experiment_failed=0
    
    # Prepare no-pbar flag
    local pbar_flag=""
    if [ $use_no_pbar -eq 1 ]; then
        pbar_flag="--no-pbar"
    fi
    if [ -n "$test_version" ]; then
        set_test_version="--test-version $test_version"
    fi 

    # Train model
    local train_cmd="$THIS_DIR/train_model.py --hash $exp_hash $set_test_version $pbar_flag"
    if [ $dryrun -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..." >> "$terminal_output"
        $train_cmd >> "$terminal_output" 2>&1
        local train_status=$?
    else
        print_exec $train_cmd
        local train_status=0
    fi
    
    if [ $train_status -ne 0 ] && [ $dryrun -eq 0 ]; then
        echo "Training failed for experiment $exp_num"
        log_experiment $exp_num $total_exp $exp_hash $test_version "TRAIN_FAILED"
        return 1
    fi
    
    # Test model (only if training succeeded)
    if [ $experiment_failed -eq 0 ]; then
        local test_cmd="$THIS_DIR/test_model.py --hash $exp_hash $set_test_version --epoch best $pbar_flag"
        if [ $dryrun -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting testing..." >> "$terminal_output"
            $test_cmd >> "$terminal_output" 2>&1
            local test_status=$?
        else
            print_exec $test_cmd
            local test_status=0
        fi
        
        if [ $test_status -ne 0 ] && [ $dryrun -eq 0 ]; then
            echo "Testing failed for experiment $exp_num"
            log_experiment $exp_num $total_exp $exp_hash $test_version "TEST_FAILED"
            return 1
        fi
    fi
    
    # Extract results (only if previous steps succeeded)
    if [ $experiment_failed -eq 0 ]; then
        local extract_cmd="$THIS_DIR/extract_rslt_statistics.py --hash $exp_hash  $set_test_version --epoch best"
        if [ $dryrun -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracting results..." >> "$terminal_output"
            $extract_cmd >> "$terminal_output" 2>&1
            local extract_status=$?
        else
            print_exec $extract_cmd
            local extract_status=0
        fi
        
        if [ $extract_status -ne 0 ] && [ $dryrun -eq 0 ]; then
            echo "Results extraction failed for experiment $exp_num"
            log_experiment $exp_num $total_exp $exp_hash $test_version "EXTRACT_FAILED"
            return 1
        else
            log_experiment $exp_num $total_exp $exp_hash $test_version "COMPLETED"
        fi
    fi
    
    if [ $dryrun -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment completed successfully" >> "$terminal_output"
    fi
    
    return 0
}

# Run the parameter sweep
experiment_num=0
failed_experiments=0

for WITH_LOGIT in "${WITH_LOGITS[@]}"; do
for TEST_VERSION in "${TEST_VERSIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for LAYERS in "${LAYERS_LIST[@]}"; do
            for NUM_GRIDS in "${NUM_GRIDS_LIST[@]}"; do
                for GRID_MIN in "${GRID_MIN_LIST[@]}"; do
                    for GRID_MAX in "${GRID_MAX_LIST[@]}"; do
                        for SCALE in "${SCALE_LIST[@]}"; do
                            for MODE in "${MODES[@]}"; do
                                for RESIDUAL in "${RESIDUALS[@]}"; do
                                    for DYNAMIC in "${DYNAMICS[@]}"; do
                                        for NO_NORMALIZE in "${NO_NORMALIZES[@]}"; do
                                            for DROPOUT in "${DROPOUTS[@]}"; do
                                                for EPOCHS in "${EPOCHS_LIST[@]}"; do
                                                    for PATIENCE in "${PATIENCE_LIST[@]}"; do
                                                        for BATCH in "${BATCH_SIZES[@]}"; do
                                                            for LR in "${LEARNING_RATES[@]}"; do
                                                                for LR_FACTOR in "${LR_FACTORS[@]}"; do
                                                                    for LR_PATIENCE in "${LR_PATIENCE_LIST[@]}"; do
                                                                        for OPTIMIZER in "${OPTIMIZERS[@]}"; do
                                                                            for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                                                                                for MOMENTUM in "${MOMENTUMS[@]}"; do
                                                                                    ((experiment_num++))
                                                                                    
                                                                                    if [ $max_experiments -gt 0 ] && [ $experiment_num -gt $max_experiments ]; then
                                                                                        echo "Reached maximum experiments limit ($max_experiments)"
                                                                                        break 20  # Break out of all loops
                                                                                    fi

                                                                            # Build configuration arguments
                                                                            CONFIGS=""
                                                                            if [ $LAYERS ]; then 
                                                                                CONFIGS="$CONFIGS --layers $LAYERS"
                                                                            fi
                                                                            CONFIGS="$CONFIGS --num-grids $NUM_GRIDS"
                                                                            CONFIGS="$CONFIGS --grid-min $GRID_MIN"
                                                                            CONFIGS="$CONFIGS --grid-max $GRID_MAX"
                                                                            CONFIGS="$CONFIGS --scale $SCALE"
                                                                            CONFIGS="$CONFIGS --mode $MODE"
                                                                            
                                                                            if [[ "$RESIDUAL" -gt 0 ]]; then
                                                                                CONFIGS="$CONFIGS --residual"
                                                                            fi 
                                                                            if [[ "$DYNAMIC" -gt 0 ]]; then
                                                                                CONFIGS="$CONFIGS --dynamic"
                                                                            fi
                                                                            if [[ "$NO_NORMALIZE" -gt 0 ]]; then
                                                                                CONFIGS="$CONFIGS --no-normalize"
                                                                            fi
                                                                            CONFIGS="$CONFIGS --dropout $DROPOUT"
                                                                            
                                                                            CONFIGS="$CONFIGS --epochs $EPOCHS"
                                                                            CONFIGS="$CONFIGS --patience $PATIENCE"
                                                                            CONFIGS="$CONFIGS --batch $BATCH"
                                                                            CONFIGS="$CONFIGS --lr $LR"
                                                                            
                                                                            if [ -n "$LR_FACTOR" ]; then
                                                                                CONFIGS="$CONFIGS --lr-factor $LR_FACTOR"
                                                                            fi
                                                                            if [ -n "$LR_PATIENCE" ]; then
                                                                                CONFIGS="$CONFIGS --lr-patience $LR_PATIENCE"
                                                                            fi
                                                                            
                                                                            CONFIGS="$CONFIGS --optimizer $OPTIMIZER"
                                                                            CONFIGS="$CONFIGS --weight-decay $WEIGHT_DECAY"
                                                                            
                                                                            if [ -n "$MOMENTUM" ]; then
                                                                                CONFIGS="$CONFIGS --momentum $MOMENTUM"
                                                                            fi
                                                                            CONFIGS="$CONFIGS --seed $SEED"
                                                                            
                                                                            if [ -n "$TEST_VERSION" ]; then
                                                                                CONFIGS="$CONFIGS --test-version $TEST_VERSION"
                                                                            fi 

                                                                            # Generate configuration hash
                                                                            if [ $WITH_LOGIT -ge 1 ]; then
                                                                                create_config_cmd="$THIS_DIR/create_configs_logits.py $CONFIGS --export --hash"
                                                                            else
                                                                                create_config_cmd="$THIS_DIR/create_configs.py $CONFIGS --export --hash"
                                                                            fi
                                                                            print_verbose [EXEC] $create_config_cmd
                                                                            exp_hash=$(dry_run $create_config_cmd)

                                                                            if [ $dryrun -ge 1 ]; then
                                                                                exp_hash="dummy_hash_${experiment_num}"
                                                                            fi

                                                                            print_verbose "[INFO] Configuration Hash: $exp_hash"

                                                                            # Save hyperparameters to CSV
                                                                            if [ $dryrun -eq 0 ]; then
                                                                                # Determine test directory path
                                                                                if [ -n "$TEST_VERSION" ]; then
                                                                                    test_dir_name="test_${TEST_VERSION}"
                                                                                else
                                                                                    test_dir_name="test_0"
                                                                                fi
                                                                                config_dir="$THIS_DIR/train/${exp_hash}/${test_dir_name}/config"
                                                                                
                                                                                # Ensure config directory exists (should have been created by create_configs.py)
                                                                                if [ -d "$config_dir" ]; then
                                                                                    # Create hyperparameters CSV
                                                                                    hyperparams_file="${config_dir}/hyperparameters.csv"
                                                                                    
                                                                                    # Write CSV header and data
                                                                                    echo "parameter,value" > "$hyperparams_file"
                                                                                    echo "with_logits,$WITH_LOGIT" >> "$hyperparams_file"
                                                                                    echo "layers,\"$LAYERS\"" >> "$hyperparams_file"
                                                                                    echo "num_grids,\"$NUM_GRIDS\"" >> "$hyperparams_file"
                                                                                    echo "grid_min,\"$GRID_MIN\"" >> "$hyperparams_file"
                                                                                    echo "grid_max,\"$GRID_MAX\"" >> "$hyperparams_file"
                                                                                    echo "scale,\"$SCALE\"" >> "$hyperparams_file"
                                                                                    echo "mode,$MODE" >> "$hyperparams_file"
                                                                                    echo "residual,$RESIDUAL" >> "$hyperparams_file"
                                                                                    echo "dynamic,$DYNAMIC" >> "$hyperparams_file"
                                                                                    echo "no_normalize,$NO_NORMALIZE" >> "$hyperparams_file"
                                                                                    echo "dropout,$DROPOUT" >> "$hyperparams_file"
                                                                                    echo "epochs,$EPOCHS" >> "$hyperparams_file"
                                                                                    echo "patience,$PATIENCE" >> "$hyperparams_file"
                                                                                    echo "batch_size,$BATCH" >> "$hyperparams_file"
                                                                                    echo "learning_rate,$LR" >> "$hyperparams_file"
                                                                                    echo "lr_factor,$LR_FACTOR" >> "$hyperparams_file"
                                                                                    echo "lr_patience,$LR_PATIENCE" >> "$hyperparams_file"
                                                                                    echo "optimizer,$OPTIMIZER" >> "$hyperparams_file"
                                                                                    echo "weight_decay,$WEIGHT_DECAY" >> "$hyperparams_file"
                                                                                    echo "momentum,$MOMENTUM" >> "$hyperparams_file"
                                                                                    echo "seed,$SEED" >> "$hyperparams_file"
                                                                                    echo "test_version,$TEST_VERSION" >> "$hyperparams_file"
                                                                                    echo "experiment_number,$experiment_num" >> "$hyperparams_file"
                                                                                    echo "config_hash,$exp_hash" >> "$hyperparams_file"
                                                                                else
                                                                                    echo "Warning: Config directory not found at $config_dir, skipping hyperparameters.csv"
                                                                                fi
                                                                            fi

                                                                            # Wait for available job slot if running parallel
                                                                            if [ $MAX_PARALLEL_JOBS -gt 1 ]; then
                                                                                wait_for_jobs $MAX_PARALLEL_JOBS
                                                                            fi

                                                                            # Run experiment (in background if parallel)
                                                                            if [ $MAX_PARALLEL_JOBS -gt 1 ]; then
                                                                                run_experiment $experiment_num $total_experiments "$exp_hash" "$CONFIGS" $WITH_LOGIT "$TEST_VERSION" $no_pbar &
                                                                            else
                                                                                run_experiment $experiment_num $total_experiments "$exp_hash" "$CONFIGS" $WITH_LOGIT "$TEST_VERSION" $no_pbar
                                                                                if [ $? -ne 0 ]; then
                                                                                    ((failed_experiments++))
                                                                                fi
                                                                            fi

                                                                            # Progress update (for sequential mode)
                                                                            if [ $MAX_PARALLEL_JOBS -eq 1 ]; then
                                                                                completed_experiments=$((experiment_num - failed_experiments))
                                                                                echo ""
                                                                                echo "Progress: $experiment_num/$total_experiments experiments completed"
                                                                                echo "Success: $completed_experiments, Failed: $failed_experiments"
                                                                                echo ""
                                                                            fi
                                                                        done
                                                                    done
                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
done

# Wait for all background jobs to complete (if running in parallel)
if [ $MAX_PARALLEL_JOBS -gt 1 ]; then
    echo ""
    echo "Waiting for all parallel jobs to complete..."
    echo "Total submitted: $experiment_num jobs"
    
    # Monitor progress
    while [ $(jobs -r | wc -l) -gt 0 ]; do
        running=$(jobs -r | wc -l)
        echo "[$(date +%H:%M:%S)] Jobs still running: $running"
        sleep 10
    done
    
    echo "All jobs completed!"
    
    # Count failed experiments from log file
    if [ $dryrun -eq 0 ] && [ -f "$RESULTS_DIR/sweep_log.txt" ]; then
        failed_experiments=$(grep -c "FAILED\|TEST_FAILED\|EXTRACT_FAILED" "$RESULTS_DIR/sweep_log.txt" || echo 0)
    fi
fi

# Final summary
echo ""
echo "========================================="
echo "Parameter Sweep Complete!"
echo "========================================="
echo "Total experiments attempted: $experiment_num"
echo "Successful experiments: $((experiment_num - failed_experiments))"
echo "Failed experiments: $failed_experiments"
echo "Results directory: $RESULTS_DIR"
echo "========================================="

if [ $dryrun -eq 0 ]; then
    echo "" >> "$RESULTS_DIR/sweep_log.txt"
    echo "Parameter sweep completed at $(date)" >> "$RESULTS_DIR/sweep_log.txt"
    echo "Total experiments attempted: $experiment_num" >> "$RESULTS_DIR/sweep_log.txt"
    echo "Successful experiments: $((experiment_num - failed_experiments))" >> "$RESULTS_DIR/sweep_log.txt"
    echo "Failed experiments: $failed_experiments" >> "$RESULTS_DIR/sweep_log.txt"
fi

if [ $failed_experiments -gt 0 ]; then
    exit 1
else
    exit 0
fi
