#!/bin/bash

# Example execution ./scripts/ablation_vgg_cifar100.sh 2.0 3 35
# Where alpha = 2.0, 3 views per image, and 0.35 CMI learner

####################################################################################
# Dataset: CIFAR-100
# Model: VGG-13
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP, EWGS, EWGS+SQAKD
# Bit-width: W1A1, W2A2, W4A4
####################################################################################
yell() { echo "$0: $*" >&2; }
die() { yell "$*"; exit 111; }
try() { "$@" || die "cannot $*"; }

set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


alpha=$1
num_transforms=$2
cmi=$3
# Note, cmi is in "centicmi"
if [ $3 -eq 0 ]; then
    teacher="fp2"
elif [ $3 -eq 20 ]; then 
    teacher="fp_cmi"
elif [ $3 -eq 35 ]; then 
    teacher="fp_cmi3"
else
    die "Unimplemented CMI: $3"
fi

echo "Distill Weight: $alpha"
echo "Number of Transforms: $num_transforms"

teacher_path="./results/CIFAR100_VGG13/${teacher}/checkpoint/last_checkpoint.pth"
echo "Teacher Path: $teacher_path"

METHOD_TYPE="t_alpha_${alpha}_trans_${num_transforms}_cmi_${cmi}"
echo "Method Type: $METHOD_TYPE"

# Logic  

python3 train_quant.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_quant' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --weight_decay 5e-4 \
                    --optimizer_m 'Adam' \
                    --optimizer_q 'Adam' \
                    --lr_m 5e-4 \
                    --lr_q 5e-6 \
                    --lr_scheduler_m 'cosine' \
                    --lr_scheduler_q 'cosine' \
                    --epochs 200 \
                    --weight_levels 4 \
                    --act_levels 4 \
                    --baseline False \
                    --use_hessian True \
                    --load_pretrain True \
                    --pretrain_path $teacher_path \
                    --log_dir './results/CIFAR100_VGG13/Ablation/'$METHOD_TYPE \
                    --distill 'siam' \
                    --teacher_arch 'vgg13_bn_fp' \
                    --teacher_path $teacher_path \
                    --seed 20240913 \
                    --num_transforms $num_transforms \
                    --transform "trivial" \
                    --kd_gamma 1.0 \
                    --kd_alpha $alpha \
                    --kd_beta 0.0 \


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"