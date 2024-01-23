import os
import torch
import argparse

# TODO add more options to backbone
# TODO look into aggregation options

def parse_arguments():
    parser = argparse.ArgumentParser(description="CNN pruning for VPR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # run
    parser.add_argument("--run_type", type=str, default=None,
                        help="Type of run to perform", choices=["tpe", "t", "tp", "te", "p", "pe", "e" ])

    # network
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backend of the network", choices=["resnet", "vgg"])
    parser.add_argument("--aggregation", type=str, default=None,
                        help="Network aggregation layer", choices=["netvlad", "gem"])

    # dataset
    parser.add_argument("--dataset_train_name", type=str, default=None,
                        help="Name of dataset used in training")
    parser.add_argument("--dataset_test_name", type=str, default=None,
                        help="Name of dataset used for testing")
    parser.add_argument("--datasets_folder", type=str, default=None,
                        help="path to datasets folder")

    # pruning
    parser.add_argument("--pruning_method", type=str, default=None,
                        help="Type of pruning to perform on the network", choices=["group_norm", "growing_reg"])
    parser.add_argument("--max_sparsity", type=float, default=None,
                        help="The level of sparsity you want to go to")
    parser.add_argument("--pruning_step", type=float, default=None,
                        help="How much to prune each pass")

    args = parser.parse_args()

    return args
