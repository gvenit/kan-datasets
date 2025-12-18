#! /bin/bash

########################################
# Configuration arguments
# -- Leave empty for default values
########################################
TEST_VERSION=               # TestLoss  -linearly_normalized
SEED=42

LAYERS="256 256"
NUM_GRIDS="4"               #  4
GRID_MIN=-1.25              # -1.2
GRID_MAX=0.25               #  0.25
SCALE=2                     #  2

EPOCHS=1000
BATCH=128
LR=3e-4
OPTIMIZER="RMSprop"         # Adam RMSprop
WEIGHT_DECAY=5e-4           # 1e-4
MOMENTUM=                   # 0.9
MODE='RSWAFF'               # 'RSWAFF'
RESIDUAL=                   # 0 / 1

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0

img_hash=
tr1_hash=

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
    echo "      --residual             Set the residual flag"
    echo "      --img                  If specified, the hash value of the image encoder/decoder model configuration"
    echo "      --tr-1                 If specified, the hash value of the first training"
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
        --img)
            img_hash=$2
            shift 
            shift ;;
        --tr-1)
            tr1_hash=$2
            shift 
            shift ;;
        --residual)
            RESIDUAL=1
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

if [ -z $img_hash ]; then
    CONFIGS=""
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

    img_hash="$THIS_DIR/create_img_enc_dec.py $CONFIGS --hash --export"
    print_verbose [EXEC] $img_hash
    img_hash=$(dry_run $img_hash)
    print_verbose $img_hash
fi

if [ -z $tr1_hash ]; then
    CONFIGS=""
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
    if [ -n "$MOMENTUM" ]; then
        CONFIGS="$CONFIGS --momentum $MOMENTUM"
    fi

    if [ -n "$SEED" ]; then
        CONFIGS="$CONFIGS --seed $SEED"
    fi
    if [ -n "$TEST_VERSION" ]; then
        CONFIGS="$CONFIGS --test-version $TEST_VERSION"
    fi 

    tr1_hash="$THIS_DIR/create_training_1.py $CONFIGS --hash --export"
    print_verbose [EXEC] $tr1_hash
    tr1_hash=$(dry_run $tr1_hash)
    print_verbose $tr1_hash
fi

if [ $dryrun -ge 1 ]; then
    test_dir=path/to/test/directory
fi

CONFIGS="-t $tr1_hash -m $img_hash"

if [ -n "$TEST_VERSION" ]; then
    CONFIGS="$CONFIGS --test-version $TEST_VERSION"
fi 
print_exec $THIS_DIR/train_img_enc_dec.py  $CONFIGS

# print_exec $THIS_DIR/test_model.py -d $test_dir

# print_exec $THIS_DIR/extract_rslt_statistics.py -d $test_dir