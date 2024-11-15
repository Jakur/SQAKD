#!/bin/bash


####################################################################################
# Dataset: CIFAR-100
# Model: VGG-13
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FP, EWGS, EWGS+SQAKD
# Bit-width: W1A1, W2A2, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE



# FP
if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --seed 20240913 \
                    --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE

elif [ $METHOD_TYPE == "fp_cmi/" ]
then 
    python3 train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --seed 20240913 \
                    --use_cmi True \
                    --cmi_weight 0.20 \
                    --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE

elif [ $METHOD_TYPE == "fp_cmi2/" ]
then 
    python3 train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --seed 20240913 \
                    --use_cmi True \
                    --cmi_weight 0.50 \
                    --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE

elif [ $METHOD_TYPE == "fp_self6/" ]
then 
    python3 train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 256 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --seed 20240913 \
                    --use_cmi True \
                    --cmi_weight 0.35 \
                    --self_supervised True \
                    --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE


elif [ $METHOD_TYPE == "foo/" ]
then
    python3 linear_head.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'vgg13_bn_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --seed 20240913 \
                    --load_pretrain True \
                    --pretrain_path './results/CIFAR100_VGG13/fp_self5/checkpoint/best_checkpoint.pth' \
                    # --pretrain_path '/home/justin/Code/Oakland/distill/mix-bt/results/CIFAR100_VGG13/2/best_checkpoint.pth' 

# './results/CIFAR100_VGG13/fp_self/checkpoint/best_checkpoint.pth'


# ===== W2A2
# EWGS
elif [ $METHOD_TYPE == "EWGS/W2A2/" ] 
then
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
                        --epochs 720 \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE


# ===== W2A2
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W2A2/" ] 
then
    python3 train_quant.py --gpu_id '1' \
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
                        --epochs 720 \
                        --weight_levels 4 \
                        --act_levels 4 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0


elif [ $METHOD_TYPE == "vid_test" ] 
then
    python3 train_quant.py --gpu_id '1' \
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
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'vid' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0

elif [ $METHOD_TYPE == "crd25/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 1.25 \
                        --all_layers 'True'

elif [ $METHOD_TYPE == "crd26/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.75 \
                        --all_layers 'True'

elif [ $METHOD_TYPE == "crd4_all/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8 \
                        --all_layers 'True'

elif [ $METHOD_TYPE == "kd_new2/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0 \
                        --all_layers 'True'

elif [ $METHOD_TYPE == "vid_new/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'vid' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp_cmi3/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 1.0 \

elif [ $METHOD_TYPE == "reverse2/" ] 
then
    python3 train_quant.py --gpu_id '0' \
                        --dataset 'cifar100' \
                        --arch 'vgg13_bn_fp' \
                        --num_workers 8 \
                        --batch_size 64 \
                        --weight_decay 5e-4 \
                        --optimizer_m 'Adam' \
                        --optimizer_q 'Adam' \
                        --lr_m 5e-4 \
                        --lr_q 5e-6 \
                        --lr_scheduler_m 'cosine' \
                        --lr_scheduler_q 'cosine' \
                        --epochs 720 \
                        --baseline False \
                        --quan_method None \
                        --load_pretrain False \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_quant' \
                        --teacher_path './results/CIFAR100_VGG13/crd4/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8 \

elif [ $METHOD_TYPE == "reverse_quant/" ] 
then
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
                        --pretrain_path './results/CIFAR100_VGG13/reverse2/checkpoint/last_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'crd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/reverse2/checkpoint/last_checkpoint.pth' \
                        --seed 20240913 \
                        --kd_gamma 1.0 \
                        --kd_alpha 0.0 \
                        --kd_beta 0.8 \
                        --all_layers 'False'

# CIFAR/results/CIFAR100_VGG13/reverse/checkpoint/last_checkpoint.pth
                         
# ===== W1A1
# EWGS
elif [ $METHOD_TYPE == "EWGS/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '1' \
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
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE

# ===== W1A1
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W1A1/" ] 
then
    python3 train_quant.py --gpu_id '2' \
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
                        --epochs 720 \
                        --weight_levels 2 \
                        --act_levels 2 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0


# ==== W4A4
# EWGS
elif [ $METHOD_TYPE == "EWGS/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '2' \
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
                        --epochs 720 \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE


# ==== W4A4
# EWGS + SQAKD
elif [ $METHOD_TYPE == "EWGS+SQAKD/W4A4/" ] 
then
    python3 train_quant.py --gpu_id '3' \
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
                        --epochs 720 \
                        --weight_levels 16 \
                        --act_levels 16 \
                        --baseline False \
                        --use_hessian True \
                        --load_pretrain True \
                        --pretrain_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --log_dir './results/CIFAR100_VGG13/'$METHOD_TYPE \
                        --distill 'kd' \
                        --teacher_arch 'vgg13_bn_fp' \
                        --teacher_path './results/CIFAR100_VGG13/fp/checkpoint/best_checkpoint.pth' \
                        --kd_gamma 0.0 \
                        --kd_alpha 1.0 \
                        --kd_beta 0.0

fi



# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"
