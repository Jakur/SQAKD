#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: VGG-13
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: EWGS + other distillation
# Bit-width: W2A2
# EWGS + SQAKD: wihout labels (gammaa=0.0)
# EWGS + other distillation: with labels (gamma = 1.0)
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

gpu_id=$1
kd_method=$2
ours=$3
if [[ $baseline == "f" ]]
then
    transform='none'
    cutmix=False
else
    transform="trivial"
    cutmix=True
fi

METHOD_TYPE="${kd_method}_${ours}"

epochs=200
quantize=4
num_workers=12
teacher='./results/CIFAR100_VGG13/fp_cutmix/checkpoint/last_checkpoint.pth'
echo $METHOD_TYPE



# EWGS + SQAKD
if [[ $kd_method == "sqakd" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0



# EWGS + AT
elif [[ $kd_method == "at" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'attention' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1000

# EWGS + NST
elif [[ $kd_method == "nst" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'nst' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50

# EWGS + SP
elif [[ $kd_method == "sp" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'similarity' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 3000


# EWGS + RKD
elif [[ $kd_method == "rkd" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'rkd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1


# EWGS + CRD
elif [[ $kd_method == "crd" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8


# EWGS + FitNet
elif [[ $kd_method == "fitnet" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'hint' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 100


# EWGS + CC
elif [[ $kd_method == "cc" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'correlation' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.02

# EWGS + VID
elif [[ $kd_method == "vid" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'vid' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0

# EWGS + FSP
elif [[ $kd_method == "fsp" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'fsp' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 50

# EWGS + FT
elif [[ $kd_method == "ft" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'factor' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 200


# EWGS + CKTF
elif [[ $kd_method == "cktf" ]] 
then
    python train_quant.py --gpu_id $gpu_id \
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
                        --epochs $epochs \
                        --weight_levels $quantize \
                        --act_levels $quantize \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path $teacher \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crdst' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path $teacher \
                        --transform $transform \
                        --cutmix $cutmix \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0 \
                        --kd_theta 0.8 \
                        --nce_k 4096
fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"
