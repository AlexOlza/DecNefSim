#!/bin/bash

# Author: aolza
# Conducts the 4 experiments currently considered given a target and non target classs

set -e  # Exit on error
# Parse arguments
tgt_class_idx=$1
non_tgt_class_idx=$2

./traditional_decnef_simulation.sh test 0 250 $tgt_class_idx $non_tgt_class_idx 40 40 500 0 0 # MNDAV
./traditional_decnef_simulation.sh test 0 250 $tgt_class_idx $non_tgt_class_idx 40 40 500 0 1 # MNDAV - ignore discr

./traditional_decnef_simulation.sh test 0 250 $tgt_class_idx $non_tgt_class_idx 40 40 500 1 0 # MNDAVMem 
./traditional_decnef_simulation.sh test 0 250 $tgt_class_idx $non_tgt_class_idx 40 40 500 1 1 # MNDAVMem - ignore discr
