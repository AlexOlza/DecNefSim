#!/bin/bash
set -e  # Exit on error
# Parse arguments
read_args=1
EXP_NAME=$1 # Output directory
trajectory_random_seed_init=$2 # The initial trajectory will be determined by this variable
n_trajectories=$3
target_class_idx=$4 # The induction target for DecNef 
non_target_class_idx=$5 # The alternative class to train the binary discriminator
lambda_inv=$6 # The inverse of lambda, which controls the subject's ability to focus
gamma_inv=$7 # The inverse of gamma, which controls the subject's ability to react to the feedback
decnef_iters=$8 # DecNef loop iterations
update_rule_idx=$9
ignore_discriminator=${10} # Whether to produce random feedback (for validation)
production=1 # Whether this execution is a trial or a definitive one

SECONDS=0
echo $read_args $EXP_NAME $trajectory_random_seed_init $n_trajectories $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production

python3 train_VAE.py --read_args $read_args $EXP_NAME \
    --trajectory_random_seed_init $trajectory_random_seed_init \
    --n_trajectories $n_trajectories \
    --target_class_idx $target_class_idx \
    --non_target_class_idx $non_target_class_idx \
    --lambda_inv $lambda_inv \
    --gamma_inv $gamma_inv \
    --decnef_iters $decnef_iters \
    --ignore_discriminator $ignore_discriminator \
    --update_rule_idx $update_rule_idx \
    --production $production

VAE_duration=$SECONDS
echo "VAE training: $((VAE_duration / 60)) minutes and $((VAE_duration % 60)) seconds."
SECONDS=0
#python3 train_CNN.py $read_args $EXP_NAME $trajectory_random_seed_init $n_trajectories $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production


python3 train_CNN.py --read_args $read_args $EXP_NAME \
    --trajectory_random_seed_init $trajectory_random_seed_init \
    --n_trajectories $n_trajectories \
    --target_class_idx $target_class_idx \
    --non_target_class_idx $non_target_class_idx \
    --lambda_inv $lambda_inv \
    --gamma_inv $gamma_inv \
    --decnef_iters $decnef_iters \
    --ignore_discriminator $ignore_discriminator \
    --update_rule_idx $update_rule_idx \
    --production $production

CNN_duration=$SECONDS
echo "CNN training: $((CNN_duration / 60)) minutes and $((CNN_duration % 60)) seconds."
SECONDS=0
#python3 traditional_decnef_n_instances.py $read_args $EXP_NAME $trajectory_random_seed $n_trajectories $target_class_idx $non_target_class_idx $lambda_inv $gamma_inv $decnef_iters $ignore_discriminator $update_rule_idx $production


python3 traditional_decnef_n_instances.py --read_args $read_args $EXP_NAME \
    --trajectory_random_seed_init $trajectory_random_seed_init \
    --n_trajectories $n_trajectories \
    --target_class_idx $target_class_idx \
    --non_target_class_idx $non_target_class_idx \
    --lambda_inv $lambda_inv \
    --gamma_inv $gamma_inv \
    --decnef_iters $decnef_iters \
    --ignore_discriminator $ignore_discriminator \
    --update_rule_idx $update_rule_idx \
    --production $production

DECNEF_duration=$SECONDS
echo "DecNef ($((n_trajectories))): $((DECNEF_duration / 60)) minutes and $((DECNEF_duration % 60)) seconds."

total_duration=$(($CNN_duration+$VAE_duration+$DECNEF_duration))
echo "Total time: $((total_duration / 60)) minutes and $((total_duration % 60)) seconds."