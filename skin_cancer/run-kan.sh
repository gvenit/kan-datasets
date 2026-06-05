#! /bin/bash

########################################
# Configuration arguments
# -- Leave empty for default values
########################################
TEST_VERSION=""      
SEED=422
WITH_LOGITS="1"

# Model 
RESIZE="64 64"             # e.g., "14 14" to resize to 14x14 (leave empty for 28x28)
CHANNELS="3"               
LAYERS="512"               # "128"
NUM_GRIDS="4"              #  4
GRID_MIN="-0.5"            # "-0.5 -0.25"
GRID_MAX="0.25"            # "0.5 0.25"
SCALE="2.5"                #  2
MODE='RSWAFF'              # 'RSWAFF' 'LeakyReLU'
RESIDUAL=0                 # 0 / 1
DYNAMIC=0                  # 0 / 1
USE_V2=0                   # 0 / 1
NO_NORMALIZE=0             # 0 / 1  NOTE: MUST BE 0 FOR LEARNING
NO_NORMALIZE_RBF=0         # 0 / 1  Disable RBF normalization (default: enabled)

# Training
DROPOUT=0.0
LINEAR_DROPOUT=0

EPOCHS=1000
PATIENCE=50
BATCH=64

# Optimization
LR=3e-4
LR_FACTOR=0.75              # Learning rate reduction factor
LR_PATIENCE=8              # Scheduler patience
OPTIMIZER="AdamW"          # Adam RMSprop
WEIGHT_DECAY=1e-4              # 5e-4
MOMENTUM=0.9                  # 0.9

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
    echo "      --use-v2               Use V2 variant with per-input-dimension learnable parameters"
    echo "      --no-normalize         Skip batch normalization"
    echo "      --no-normalize-rbf     Skip RBF output normalization (default: enabled)"
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
        --use-v2)
            USE_V2=1
            shift ;;
        --no-normalize)
            NO_NORMALIZE=1
            shift ;;
        --no-normalize-rbf)
            NO_NORMALIZE_RBF=1
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

eval set -- "$PARAMS"

THIS_DIR=$(dirname $(realpath $0))
print_exec cd $(dirname $THIS_DIR)

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
    if [[ -n "$RESIZE" ]]; then
        CONFIGS="$CONFIGS --resize $RESIZE"
    fi
    if [[ -n "$DROPOUT" ]] && [[ "$DROPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout $DROPOUT"
    fi 
    if [[ -n "$LINEAR_DROPOUT" ]] && [[ "$LINEAR_DROPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout-linear $LINEAR_DROPOUT"
    fi 
    if [ -n "$TEST_VERSION" ]; then
        CONFIGS="$CONFIGS --test-version $TEST_VERSION"
    fi 
    img_hash="$THIS_DIR/create_kan_enc_dec.py $CONFIGS --hash --export"
    print_verbose [EXEC] $img_hash
    img_hash=$(dry_run $img_hash)
fi
print_verbose [INFO] Image Encoder-Decoder Hash: $img_hash

if [ -z $tr1_hash ]; then
    CONFIGS=""
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

    if [ $WITH_LOGITS -ge 1 ]; then
        tr1_hash="$THIS_DIR/create_tr_1_logits.py $CONFIGS --export --hash"
    else
        tr1_hash="$THIS_DIR/create_tr_1.py $CONFIGS --export --hash"
    fi
    print_verbose [EXEC] $tr1_hash
    tr1_hash=$(dry_run $tr1_hash)
fi
print_verbose [INFO] Training Stage 1 Hash: $tr1_hash

if [ $dryrun -ge 1 ]; then
    test_dir=path/to/test/directory
fi

CONFIGS="-t $tr1_hash -m $img_hash"

if [ -n "$TEST_VERSION" ]; then
    CONFIGS="$CONFIGS --test-version $TEST_VERSION"
fi 
# print_exec $THIS_DIR/train_model.py $CONFIGS

# print_exec $THIS_DIR/test_model.py  $CONFIGS --epoch best

# print_exec $THIS_DIR/extract_rslt_statistics.py $CONFIGS --epoch best
