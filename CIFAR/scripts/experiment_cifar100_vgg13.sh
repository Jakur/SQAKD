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

quantization=4
alpha=2.0
num_epochs=200
num_transforms=1
teacher="fp_cutmix"
seed=20250215

echo "Distill Weight: $alpha"
echo "Number of Transforms: $num_transforms"

teacher_path="./results/CIFAR100_VGG13/${teacher}/checkpoint/last_checkpoint.pth"
echo "Teacher Path: $teacher_path"

METHOD_TYPE="${quantization}_${transform}_${cutmix}_${num_transforms}"
echo "Method Type: $METHOD_TYPE"

# Logic  

python3 train_quant.py --gpu_id $gpu_id \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_quant' \
                    --num_workers $num_workers \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \
                    --lr_m 5e-4 \
                    --lr_q 5e-6 \
                    --lr_scheduler_m 'cosine' \
                    --lr_scheduler_q 'cosine' \
                    --epochs $num_epochs \
                    --weight_levels $quantization \
                    --act_levels $quantization \
                    --baseline False \
                    --use_hessian True \
                    --load_pretrain True \
                    --pretrain_path $teacher_path \
                    --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                    --distill 'siam' \
                    --teacher_arch 'vgg13_bn_fp' \
                    --teacher_path $teacher_path \
                    --seed $seed \
                    --num_transforms $num_transforms \
                    --transform $transform \
                    --cutmix $cutmix \
                    --kd_gamma 1.0 \
                    --kd_alpha $alpha \
                    --decay_alpha False \
                    --kd_beta 0.0 \


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"