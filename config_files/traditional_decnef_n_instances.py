#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:30:15 2025

@author: alexolza
"""

import argparse
update_rule_names = ['MNDAV', 'MNDAVMem']
def traditional_decnef_n_instances_parser():
    parser = argparse.ArgumentParser(
                        prog='traditional_decnef_n_instances',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    parser.add_argument('--read_args', type= int, default=1)
    parser.add_argument('EXP_NAME')
    parser.add_argument('--trajectory_random_seed_init', required = False, default= 0, type= int)
    parser.add_argument('--n_trajectories', required = False, default= 10, type= int)
    parser.add_argument('--target_class_idx', required = False, default= 0, type= int)
    parser.add_argument('--non_target_class_idx', required = False, default= 1, type= int)
    parser.add_argument('--lambda_inv', required = False, default= 40, type= int)
    parser.add_argument('--gamma_inv', required = False, default= 40, type= int)
    parser.add_argument('--decnef_iters', required = False, default= 500, type= int)
    parser.add_argument('--ignore_discriminator', required = False, default= 0, type= int)
    parser.add_argument('--update_rule_idx', required = False, default= 0, type= int)
    parser.add_argument('--production', required = False, default= 1, type= int)
    parser.add_argument('--generator_name', type= str, required = False, default='VAE')
    parser.add_argument('--discriminator_type', type= str, required = False, default='CNN')

    c0 = parser.parse_args()
    parser.add_argument('--update_rule_name', type= str, required = False, default=update_rule_names[c0.update_rule_idx])
    # parser.parse_args(args=['--update_rule_name', update_rule_names[c0.update_rule_idx]], namespace=c0)
    config = parser.parse_args()
    return config
