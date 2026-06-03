#! /bin/bash

########################################
# Configuration arguments
# -- Leave empty for default values
########################################
TEST_VERSION=""      # TestLoss  -linearly_normalized
SEED=42

RESIZE=                     # e.g., "16 16" to resize to 16x16 (leave empty for 32x32)

HIDDEN="16 8 4"                  
ENCODED="64 10"                
NUM_GRIDS="8"               #  4
GRID_MIN="-1.0"        # "-0.5 -0.25"            
GRID_MAX="2.0"          # "0.5 0.25"             
SCALE="4.0"                   #  2
MODE='RSWAFF'               # 'RSWAFF' 'LeakyReLU'
RESIDUAL=                   # 0 / 1
DYNAMIC=                    # 0 / 1
NO_NORMALIZE=1              # 0 / 1
DROPOUT=0.25

EPOCHS=10000
PATIENCE=50
BATCH=16                  
LR=5e-4
LR_FACTOR=0.5              # Learning rate reduction factor
LR_PATIENCE=15              # Scheduler patience
OPTIMIZER="AdamW"          # Adam RMSprop
WEIGHT_DECAY=5e-4              # 5e-4
MOMENTUM=                  # 0.9

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0
exp_hash=

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
usage () {
    echo NAME
    echo "      run.sh [-h] [-d] [-v] [-p]" 
    echo
    echo DESCRIPTION
    echo "      Compiles and executes the simutalion for the proviced testbench."
    echo
    echo Parameters
    echo "      -h                     Prints out help"
    echo "      -s, --seed             Change the seed"
    echo "      -d, --dryrun           Dry run of the script"
    echo "      -v, --verbose          Prints the to be executed commands"
    echo "      -p, --purge            Purges any existing output files before generating them"
    echo "      --residual             Set the residual flag"
    echo "      --dynamic              Set the dynamic flag"
    echo "      --no-normalize         Skip batch normalization"
    echo "      --hash                 The hash value of the configuration to use"
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
            update=""
            shift ;;
        -s|--seed)
            SEED=$2
            shift 
            shift ;;
        --residual)
            RESIDUAL=1
            shift ;;
        --dynamic)
            DYNAMIC=1
            shift ;;
        --no-normalize)
            NO_NORMALIZE=1
            shift ;;
        --hash)
            exp_hash=$2
            shift 
            shift ;;
        -*|--*=)  # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            exit 1 ;;
        *)  # preserve positional arguments
            PARAMS="$PARAMS $1"
            shift ;;
    esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

THIS_DIR=$(dirname $(realpath $0))
print_exec cd $(dirname $THIS_DIR)

if [ ! -n "$exp_hash" ]; then
    CONFIGS=""
    if [ -n "$HIDDEN" ]; then
        CONFIGS="$CONFIGS --hidden $HIDDEN"
    fi 
    if [ -n "$ENCODED" ]; then
        CONFIGS="$CONFIGS --encoded $ENCODED"
    fi 
    if [ -n "$NUM_GRIDS" ]; then
        CONFIGS="$CONFIGS --num-grids $NUM_GRIDS"
    fi 
    if [ -n "$GRID_MIN" ]; then
        CONFIGS="$CONFIGS --grid-min $GRID_MIN"
    fi 
    if [ -n "$GRID_MAX" ]; then
        CONFIGS="$CONFIGS --grid-max $GRID_MAX"
    fi 
    if [ -n "$SCALE" ]; then
        CONFIGS="$CONFIGS --scale $SCALE"
    fi 

    if [ -n "$MODE" ]; then
        CONFIGS="$CONFIGS --mode $MODE"
    fi 
    if [[ -n "$RESIDUAL" ]] && [[ "$RESIDUAL" -gt 0 ]]; then
        CONFIGS="$CONFIGS --residual"
    fi 
    if [[ -n "$DYNAMIC" ]] && [[ "$DYNAMIC" -gt 0 ]]; then
        CONFIGS="$CONFIGS --dynamic"
    fi
    if [[ -n "$NO_NORMALIZE" ]] && [[ "$NO_NORMALIZE" -gt 0 ]]; then
        CONFIGS="$CONFIGS --no-normalize"
    fi
    if [[ -n "$RESIZE" ]]; then
        CONFIGS="$CONFIGS --resize $RESIZE"
    fi
    if [[ -n "$DROPOUT" ]] && [[ "$DROPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout $DROPOUT"
    fi 

    if [ -n "$EPOCHS" ]; then
        CONFIGS="$CONFIGS --epochs $EPOCHS"
    fi
    if [ -n "$PATIENCE" ]; then
        CONFIGS="$CONFIGS --patience $PATIENCE"
    fi
    if [ -n "$BATCH" ]; then
        CONFIGS="$CONFIGS --batch $BATCH"
    fi
    if [ -n "$LR" ]; then
        CONFIGS="$CONFIGS --lr $LR"
    fi

    if [ -n "$LR_FACTOR" ]; then
        CONFIGS="$CONFIGS --lr-factor $LR_FACTOR"
    fi
    if [ -n "$LR_PATIENCE" ]; then
        CONFIGS="$CONFIGS --lr-patience $LR_PATIENCE"
    fi

    if [ -n "$OPTIMIZER" ]; then
        CONFIGS="$CONFIGS --optimizer $OPTIMIZER"
    fi
    if [ -n "$WEIGHT_DECAY" ]; then
        CONFIGS="$CONFIGS --weight-decay $WEIGHT_DECAY"
    fi
    if [ -n "$MOMENTUM" ]; then
        CONFIGS="$CONFIGS --momentum $MOMENTUM"
    fi

    if [ -n "$SEED" ]; then
        CONFIGS="$CONFIGS --seed $SEED"
    fi
    if [ -n "$TEST_VERSION" ]; then
        CONFIGS="$CONFIGS --test-version $TEST_VERSION"
    fi 

    test_dir="$THIS_DIR/create_imgkan_configs.py $CONFIGS --export --hash"
    print_verbose [EXEC] $test_dir
    exp_hash=$(dry_run $test_dir)

    if [ $dryrun -ge 1 ]; then
        test_dir=path/to/test/directory
    fi
fi

if [ -n "$exp_hash" ]; then
    print_verbose [INFO] Configuration Hash: $exp_hash
fi

if [ -n "$TEST_VERSION" ]; then
    set_test_version="--test-version $TEST_VERSION"
fi 

print_exec $THIS_DIR/train_model.py --hash $exp_hash $set_test_version

print_exec $THIS_DIR/test_model.py --hash $exp_hash $set_test_version --epoch best

print_exec $THIS_DIR/extract_rslt_statistics.py --hash $exp_hash $set_test_version --epoch best