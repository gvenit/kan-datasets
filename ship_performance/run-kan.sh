#! /bin/bash

########################################
# Configuration arguments
# -- Leave empty for default values
########################################
TEST_VERSION=HuberLoss-square # TestLoss
SEED=42

LAYERS="256 256 256"
NUM_GRIDS=
GRID_MIN=
GRID_MAX=
SCALE=

EPOCHS=10000
BATCH=64
LR=3e-4
OPTIMIZER= #"RMSprop"
WEIGHT_DECAY=
MODE='square'

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################

PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0

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

CONFIGS=""
if [ -n "$TEST_VERSION" ]; then
    CONFIGS="$CONFIGS --test-version $TEST_VERSION"
fi 
if [ -n "$SEED" ]; then
    CONFIGS="$CONFIGS --seed $SEED"
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

if [ -n "$EPOCHS" ]; then
    CONFIGS="$CONFIGS --epochs $EPOCHS"
fi
if [ -n "$BATCH" ]; then
    CONFIGS="$CONFIGS --batch $BATCH"
fi
if [ -n "$LR" ]; then
    CONFIGS="$CONFIGS --lr $LR"
fi


if [ -n "$OPTIMIZER" ]; then
    CONFIGS="$CONFIGS --optimizer $OPTIMIZER"
fi
if [ -n "$WEIGHT_DECAY" ]; then
    CONFIGS="$CONFIGS --weight-decay $WEIGHT_DECAY"
fi

print_verbose [EXEC] $THIS_DIR/create_configs.py $CONFIGS --export
test_dir=$(dry_run $THIS_DIR/create_configs.py $CONFIGS --export)

if [ $dryrun -ge 1 ]; then
    test_dir=path/to/test/directory
fi

print_exec $THIS_DIR/train_model.py -d $test_dir

print_exec $THIS_DIR/test_model.py -d $test_dir

print_exec $THIS_DIR/extract_rslt_statistics.py -d $test_dir