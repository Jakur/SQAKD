#!/bin/bash

yell() { echo "$0: $*" >&2; }
die() { yell "$*"; exit 111; }
try() { "$@" || die "cannot $*"; }

set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

gpu_id=$1
num_workers=$2
transform=$3
cutmix=$4

q_bits=3
q_method="pact"

echo "Distill Weight: $alpha"
echo "Number of Transforms: $num_transforms"

path_root="./results/tiny-imagenet_efficient"
METHOD_TYPE="${q_method}_w${q_bits}a${q_bits}_${transform}_${cutmix}"
echo "Method Type: $METHOD_TYPE"

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --save_root_path "${path_root}/$METHOD_TYPE" \
    -a efficientnet_b0 \
    --batch-size 64 \
    --loss-scale 128.0 \
    --workers $num_workers \
    --optimizer_type 'SGD' \
    --lr 0.004 \
    --weight_decay 1e-4 \
    --backward_method "org" \
    /home/users/kzhao27/tiny-imagenet-200 \
    --load_pretrain \
    --pretrain_path "${path_root}/fp/checkpoints/checkpoint.pth.tar" \
    --distill True \
    --teacher_arch efficientnet_b0 \
    --teacher_path "${path_root}/fp/checkpoints/checkpoint.pth.tar" \
    --gamma 3.0 \
    --alpha 6.0 \
    --quantization $q_method \
    --bits $q_bits \
    --cutmix $cutmix \
    --transform $transform \
    --epochs=100

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"