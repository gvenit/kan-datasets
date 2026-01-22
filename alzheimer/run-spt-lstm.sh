#! /bin/bash

########################################
# Configuration arguments
# -- Leave empty for default values
########################################
TEST_VERSION=               # TestLoss  -linearly_normalized
SEED=42

IMG_HASH=e2dee2fa4e072bc524e2dc00a97d36c714ebcaaa
TR1_HASH=9b83599ad476f9c65c69630664962819d8025be1

LAYERS="128 128 128"                  
DROPOUT=0.15

EPOCHS=1000
PATIENCE=15
BATCH=16
LR=5e-3
OPTIMIZER="Adam"            # Adam RMSprop
WEIGHT_DECAY=5e-5           # 1e-4
MOMENTUM=                   # 0.9

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0

spt_hash=
tr2_hash=

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
    echo "      --img                  If specified, the hash value of the image encoder/decoder model configuration"
    echo "      --spt                  If specified, the hash value of the spatial encoder/decoder model configuration"
    echo "      --tr-1                 If specified, the hash value of the first training"
    echo "      --tr-2                 If specified, the hash value of the second training"
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
            IMG_HASH=$2
            shift 
            shift ;;
        --spt)
            spt_hash=$2
            shift 
            shift ;;
        --tr-1)
            TR1_HASH=$2
            shift 
            shift ;;
        --tr-2)
            tr2_hash=$2
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

if [ -z $spt_hash ]; then
    CONFIGS=""
    if [ -n "$IMG_HASH" ]; then
        CONFIGS="$CONFIGS --img $IMG_HASH"
    fi 
    if [ -n "$LAYERS" ]; then
        CONFIGS="$CONFIGS --hidden $LAYERS"
    fi 
    if [[ -n "$DROPOUT" ]] && [[ "$DROPOUT" ]]; then
        CONFIGS="$CONFIGS --dropout $DROPOUT"
    fi 
    if [ -n "$TEST_VERSION" ]; then
        CONFIGS="$CONFIGS --test-version $TEST_VERSION"
    fi 

    spt_hash="$THIS_DIR/create_lstm_spt_enc_dec.py $CONFIGS --hash --export"
    print_verbose [EXEC] $spt_hash
    spt_hash=$(dry_run $spt_hash)
fi
print_verbose [INFO] Spatial Encoder-Decoder Hash: $spt_hash
print_verbose [INFO] Image Encoder-Decoder Hash: $IMG_HASH

if [ -z $tr2_hash ]; then
    CONFIGS=""
    if [ -n "$TR1_HASH" ]; then
        CONFIGS="$CONFIGS --tr-1 $TR1_HASH"
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
    tr2_hash="$THIS_DIR/create_tr_2.py $CONFIGS --hash --export"
    print_verbose [EXEC] $tr2_hash
    tr2_hash=$(dry_run $tr2_hash)
fi
print_verbose [INFO] Training Stage 1 Hash: $TR1_HASH
print_verbose [INFO] Training Stage 2 Hash: $tr2_hash

if [ $dryrun -ge 1 ]; then
    test_dir=path/to/test/directory
fi

CONFIGS="-t $tr2_hash -m $spt_hash"

if [ -n "$TEST_VERSION" ]; then
    CONFIGS="$CONFIGS --test-version $TEST_VERSION"
fi 
print_exec $THIS_DIR/train_2.py $CONFIGS

print_exec $THIS_DIR/test_2.py  $CONFIGS --epoch best

print_exec $THIS_DIR/extract_rslt_tr_2.py $CONFIGS --epoch best