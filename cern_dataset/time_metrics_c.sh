#!/bin/bash

########################################
# DO NOT ALTER BEYOND THIS POINT
########################################
PARAMS=""  # to store positional arguments

dryrun=0
verbose=0
purge=0
force=0
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

while [ "$#" -gt 0 ] ; do
    case "$1" in
        -d|--dry-run) 
            dryrun=1
            verbose=1
            shift ;;
        -v|--verbose) 
            verbose=1
            shift ;;
        -f|--force) 
            force=1
            shift ;;
        *)  # preserve positional arguments
            PARAMS="$PARAMS $1"
            shift ;;
    esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

THIS_DIR=$(dirname $(realpath $0))
print_exec cd $(dirname $THIS_DIR)
targ_file=$(realpath "cern_dataset/time_metrics_c.exe")
targ_dir=$(dirname $targ_file)

if [[ $force -gt 0 ]] || [[ ! -e $targ_file ]]; then
    print_exec "python -m nuitka \
        -j $(nproc --all) \
        --enable-plugin=no-qt \
        --lto=yes \
        --pgo-python \
        --pgo-args=\""$*"\" \
        --module-parameter=torch-disable-jit=no \
        --nofollow-import-to=matplotlib \
        --nofollow-import-to=tkinter \
        --nofollow-import-to=torchvision \
        --nofollow-import-to=pillow \
        --nofollow-import-to=albumentationsx \
        --nofollow-import-to=pytest \
        --show-modules \
        --output-dir="$targ_dir" --output-filename=$(basename $targ_file) \
        cern_dataset/time_metrics.py 
fi

print_exec cern_dataset/time_metrics_c.exe "$*"