#!/bin/bash

# Parse arguments
read_args=1
EXP_NAME=$1 # Output directory
trajectory_random_seed=$2 # Each trajectory will be determined by this variable
target_class_idx=$3 # The induction target for DecNef 
non_target_class_idx=$4 # The alternative class to train the binary discriminator
lambda_inv=$5 # The inverse of lambda, which controls the subject's ability to focus
gamma_inv=$6 # The inverse of gamma, which controls the subject's ability to react to the feedback
decnef_iters=$7 # DecNef loop iterations
update_rule_idx=$8
ignore_discriminator=$9 # Whether to produce random feedback (for validation)
production=1 # Whether this execution is a trial or a definitive one


python3 train_VAE.py $read_args $EXP_NAME $trajectory_random_seed $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production

python3 train_CNN.py $read_args $EXP_NAME $trajectory_random_seed $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production

python3 traditional_decnef_single_instance.py $read_args $EXP_NAME $trajectory_random_seed $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production
