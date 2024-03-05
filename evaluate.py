# should loop through all models in the test folder
# use the gl to get the recall @N for each model saved in text in each folder
# should then generate a graph of recall@N against sparsity from these
# graph should be saved in test folder

from os.path import join
import numpy as np
from matplotlib import pyplot as plt
import os
import deep_visual_geo_localization_benchmark as gl


def evaluate(args, vargs):
    performance = []
    sparsity = []

    # lots of nesting!
    # is really just cuz each model will be in their own folder
    for f in os.listdir(args.run_path):
        dirpath = join(args.run_path, f)
        if os.path.isdir(dirpath):
            for g in os.listdir(dirpath):

                if os.path.isfile(join(dirpath, g)):
                    name = g.split(".")

                    if name[1] == "pth":
                        sparsity.append(int(name[0]))
                        args.resume = join(args.run_path, f, g)
                        args.save_dir = join(args.run_path, f)
                        performance.append(gl.eval(args))


    performance = np.array(performance).T

    for i, p in enumerate(performance):
        recalls = [1, 5, 10, 20]
        plt.plot(sparsity, p, label=f'recall@{recalls[i]}')

    plt.xlabel("Sparsity")
    plt.ylabel("Recall")
    plt.title(f"{args.backbone}-{args.aggregation}-{args.pruning_method}")
    plt.legend()
    plt.savefig(join(args.run_path, "recalls.pdf"))



    """
    # utilise  to get detailed benchamark for each level of sparsity generated
    def bench():
        # precompute and save the necessary outputs for the benchmark
        # these outputs should be saved in the current directory

        save_matching_info = 1  # If save_matching_info=0, save matching info in 'vpr_precomputed_matches_directory' for all the techniques in 'VPR_techniques' on the dataset specified in 'vpr_dataset_directory'.
        scale_percent = 100  # Provision for resizing (with aspect-ratio maintained) of query and reference images between 0-100%. 100% is equivalent to NO resizing.

        VPR_evaluation_mode = vargs["evalmode"]
        save_matching_info = vargs["savematchinginfo"]
        dataset_name = vargs["datasetname"]
        vpr_dataset_directory = vargs["datasetdirectory"]
        vpr_precomputed_matches_directory = vargs["precomputedmatchesdirectory"]
        VPR_techniques = vargs["VPRtechniquenames"]

        #VPR_Bench.execute_evaluation_mode.exec_eval_mode(VPR_evaluation_mode, dataset_name, vpr_dataset_directory,
        #                                                 vpr_precomputed_matches_directory, VPR_techniques,
        #                                                 save_matching_info, scale_percent)

    return
    """