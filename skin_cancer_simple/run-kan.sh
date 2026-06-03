#! /bin/bash

########################################
# Configuration arguments
########################################
TEST_VERSION=""
SEED=42
WITH_LOGITS="1"

RESIZE="" # default is 64x64 and is configured in create_configs_logits.py
CHANNELS="3"
LAYERS="512"
NUM_GRIDS="4"
GRID_MIN="-0.5"
GRID_MAX="0.5"
SCALE="2.5"
MODE='RSWAFF'
RESIDUAL=0
DYNAMIC=0
USE_V2=0
NO_NORMALIZE=1
NO_NORMALIZE_RBF=1

# Training
DROPOUT=0
LINEAR_DROPPOUT=0

EPOCHS=500
PATIENCE=50
BATCH=32

# Optimization
LR=3e-4
LR_FACTOR=0.75
LR_PATIENCE=8
OPTIMIZER="AdamW"
WEIGHT_DECAY=
MOMENTUM=0.9

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""

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
    echo "      run-kan-2.sh [-h] [-d] [-v] [-p]"
    echo
    echo DESCRIPTION
    echo "      Simplified KAN model: flattened image -> 512 hidden -> 7 logits"
    echo
    echo Parameters
    echo "      -h                     Prints out help"
    echo "      -s, --seed             Change the seed"
    echo "      -d, --dryrun           Dry run of the script"
    echo "      -v, --verbose          Prints the to be executed commands"
    echo "      -p, --purge            Purges any existing output files before generating them"
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
        --hash)
            exp_hash=$2
            shift
            shift ;;
        -*|--*=)
            echo "Error: Unsupported flag $1" >&2
            exit 1 ;;
        *)
            PARAMS="$PARAMS $1"
            shift ;;
    esac
done

eval set -- "$PARAMS"

# Enforce logits-only mode
if [ "$WITH_LOGITS" != "1" ]; then
    echo "Error: run-kan-2.sh only supports logits mode (WITH_LOGITS=1)" >&2
    exit 1
fi

THIS_DIR=$(dirname $(realpath $0))
print_exec cd $(dirname $THIS_DIR)

# Single unified config step: create_configs_logits.py handles both model + training
if [ ! -n "$exp_hash" ]; then
    CONFIGS=""
    if [[ -n "$RESIZE" ]]; then
        CONFIGS="$CONFIGS --resize $RESIZE"
    fi
    if [ -n "$LAYERS" ]; then
        CONFIGS="$CONFIGS --layers $LAYERS"
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
    if [[ -n "$USE_V2" ]] && [[ "$USE_V2" -gt 0 ]]; then
        CONFIGS="$CONFIGS --use-v2"
    fi
    if [[ -n "$NO_NORMALIZE" ]] && [[ "$NO_NORMALIZE" -gt 0 ]]; then
        CONFIGS="$CONFIGS --no-normalize"
    fi
    if [[ -n "$NO_NORMALIZE_RBF" ]] && [[ "$NO_NORMALIZE_RBF" -gt 0 ]]; then
        CONFIGS="$CONFIGS --no-normalize-rbf"
    fi
    if [[ -n "$DROPOUT" ]] && [[ "$DROPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout $DROPOUT"
    fi
    if [[ -n "$LINEAR_DROPPOUT" ]] && [[ "$LINEAR_DROPPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout-linear $LINEAR_DROPPOUT"
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

    test_dir="$THIS_DIR/create_configs_logits.py $CONFIGS --export --hash"
    print_verbose [EXEC] $test_dir
    exp_hash=$(dry_run $test_dir)
    if [ $dryrun -ge 1 ]; then
        exp_hash="path/to/config/hash"
    fi
fi

if [ -n "$exp_hash" ]; then
    print_verbose [INFO] Configuration Hash: $exp_hash
fi
set_test_version=""
if [ -n "$TEST_VERSION" ]; then
    set_test_version="--test-version $TEST_VERSION"
fi

print_exec $THIS_DIR/train_model.py --hash $exp_hash $set_test_version
print_exec $THIS_DIR/test_model.py --hash $exp_hash $set_test_version --epoch best
print_exec $THIS_DIR/extract_rslt_statistics.py --hash $exp_hash $set_test_version --epoch best
