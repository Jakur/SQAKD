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
seed=$5

quantization=4
alpha=2.0
num_epochs=400
num_transforms=1

echo "Distill Weight: $alpha"
echo "Number of Transforms: $num_transforms"

teacher_path="/workspace/c10_r20_fp_cutmix.pth"
echo "Teacher Path: $teacher_path"

METHOD_TYPE="${quantization}_${transform}_${cutmix}_${seed}"
echo "Method Type: $METHOD_TYPE"

# Logic  

python3 train_quant.py --gpu_id $gpu_id \
                    --arch 'resnet20_quant' \
                    --epochs $num_epochs \
                    --num_workers $num_workers \
                    --weight_levels $quantization \
                    --act_levels $quantization \
                    --baseline False \
                    --use_hessian True \
                    --load_pretrain True \
                    --pretrain_path $teacher_path \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --distill 'kd' \
                    --num_transforms $num_transforms \
                    --transform $transform \
                    --cutmix $cutmix \
                    --teacher_arch 'resnet20_fp' \
                    --teacher_path $teacher_path \
                    --kd_gamma 1.0 \
                    --kd_alpha $alpha \
                    --kd_beta 0.0 \
                    --seed $seed


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"