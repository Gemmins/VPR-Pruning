# should loop through all models in the test folder
# at each model it should use the vpr-bench to generate metrics for that model
# these should be saved in the model folder
# also use the gl to get the recall @N for each model saved in text in each folder
# should then generate a graph of recall@N against sparsity from these
# graph should be saved in test folder
from os.path import join

import numpy as np

import VPR_Bench
import os
import deep_visual_geo_localization_benchmark as gl
def evaluate(run_path, args):

    performance = np.empty((1, 5))
    sparsity = []

    # lost of nesting!
    # is really just cuz each model will be in their own folder
    for f in os.listdir(args.run_path):
        if os.path.isdir(f):
            for g in os.listdir(f):
                if os.path.isfile(f):
                    if f.split(".")[1] == "pth":
                        args.resume = join(args.run_path, f, g)
                        gl.eval(args)




    # utilise VPR-Bench to get detailed benchamark for each level of sparsity generated
    def bench():
        # precompute and save the necessary outputs for the benchmark
        # these outputs should be saved in the current directory


        save_matching_info = 1  # If save_matching_info=0, save matching info in 'vpr_precomputed_matches_directory' for all the techniques in 'VPR_techniques' on the dataset specified in 'vpr_dataset_directory'.
        scale_percent = 100  # Provision for resizing (with aspect-ratio maintained) of query and reference images between 0-100%. 100% is equivalent to NO resizing.

        VPR_evaluation_mode = args["evalmode"]
        save_matching_info = args["savematchinginfo"]
        dataset_name = args["datasetname"]
        vpr_dataset_directory = args["datasetdirectory"]
        vpr_precomputed_matches_directory = args["precomputedmatchesdirectory"]
        VPR_techniques = args["VPRtechniquenames"]

        VPR_Bench.execute_evaluation_mode.exec_eval_mode(VPR_evaluation_mode, dataset_name, vpr_dataset_directory,vpr_precomputed_matches_directory, VPR_techniques, save_matching_info, scale_percent)

    return
