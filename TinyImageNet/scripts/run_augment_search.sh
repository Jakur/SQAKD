#!/bin/bash


# ====================== Tiny-ImageNet, MobileNet-V2 ===============================================================================
# SGD
# batch-size 
#     PACT: 64
#     LSQ, DoReFa: 32
# lr, weight_decay, initilization
#     fp: lr: 0.004, weight_decay: 1e-4
#     Quantized: 5e-4, weight_decay: 5e-4, initilized with pretrained fp model

# Note that:
# 1. You need to choose function of "get_extra_config()" in get_config.py, based on the QAT method, including PACT, LSQ, and DoReFa
# 2. Modify the bit-widths for both weights and activations, inside the function of "get_extra_config()",
#    E.g., the setting of 4-bit is as follows: 'bit': 4
# ==================================================================================================================================

# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE


# ===== fp
if [ $METHOD_TYPE == "mobilenet_v2" ] 
then
    CUDA_VISIBLE_DEVICES=0 python augment_search.py --gpu_id '0' \
                    --teacher_arch mobilenet_v2 \
                    --seed 20240913 \
                    --teacher_path "./results/tiny-imagenet_mobilenetV2/mobilenet_v2_fp/checkpoints/checkpoint.pth.tar"

elif [ $METHOD_TYPE == "resnet18" ] 
then
    CUDA_VISIBLE_DEVICES=0 python augment_search.py --gpu_id '0' \
                --teacher_arch resnet18_imagenet \
                --seed 20240913 \
                --teacher_path "./results/tiny-imagenet/resnet18_fp/checkpoints/checkpoint.pth.tar"

elif [ $METHOD_TYPE == "cmresnet18" ] 
then
    CUDA_VISIBLE_DEVICES=0 python augment_search.py --gpu_id '0' \
                --teacher_arch resnet18_imagenet \
                --seed 20240913 \
                --teacher_path "./results/tiny-imagenet/resnet18_fp_cutmix/checkpoints/checkpoint.pth.tar"

fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"