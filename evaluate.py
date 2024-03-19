# should loop through all models in the test folder
# use the gl to get the recall @N for each model saved in text in each folder
# should then generate a graph of recall@N against sparsity from these
# graph should be saved in test folder

from os.path import join
import numpy as np
import pandas
from matplotlib import pyplot as plt
import os
import deep_visual_geo_localization_benchmark as gl
import torch
from io import BytesIO
import dataframe_image as dfi
import torch_pruning as tp
def evaluate(args, vargs):
    performance = []
    sparsity = []
    timings = []
    dimensions = []
    columns = []
    # lots of nesting!
    # is really just cuz each model will be in their own folder
    a = os.listdir(args.run_path)

    for f in os.listdir(args.run_path):
        dirpath = join(args.run_path, f)
        if os.path.isdir(dirpath):
            for g in os.listdir(dirpath):

                if os.path.isfile(join(dirpath, g)):
                    name = g.split(".")

                    if name[1] == "pth":
                        sparsity.append(float("0." + name[0]))
                        args.resume = join(args.run_path, f, g)
                        args.save_dir = join(args.run_path, f)
                        #performance.append(gl.eval(args))
                        timings.append(estimate_latency(args))
                        dimensions.append(get_dimensions(args))
                        columns.append(float("0." + name[0]))

    performance = np.array(performance).T
    eval_dir = join(args.run_path, "eval")
    os.mkdir(eval_dir)

    """
    for i, p in enumerate(performance):
        recalls = [1, 5, 10, 20]
        plt.plot(sparsity, p, label=f'recall@{recalls[i]}')
    plt.ylim(0, 100)
    plt.xlabel("Sparsity")
    plt.ylabel("Recall")
    plt.title(f"{args.backbone}-{args.aggregation}-{args.pruning_method}")
    plt.legend()
    plt.savefig(join(eval_dir, "recalls.pdf"))
    """

    timings = np.array(timings).T
    plt.figure()
    plt.plot(sparsity, timings[0], label="average time (ms)")
    above = timings[0]+timings[1]
    #plt.plot(sparsity, above, '--', label='std dev')
    plt.plot(sparsity, (timings[0] + timings[1]), '--', label='std dev', color='r')
    plt.plot(sparsity, (timings[0] - timings[1]), '--', color='r')
    plt.legend()
    plt.xlabel("Sparsity")
    plt.ylabel("Inference time")
    plt.title(f"{args.backbone}-{args.aggregation}-{args.pruning_method}")
    plt.savefig(join(eval_dir, "timing.pdf"))

    np.savetxt(join(eval_dir, "recalls.csv"), performance, delimiter=",")

    np.savetxt(join(eval_dir, "timing.csv"), timings, delimiter=",")

    indexes = []
    for i in range(len(dimensions[0])):
        indexes.append(f"conv layer:{i}")

    df = pandas.DataFrame(data=np.array(dimensions).T, index=indexes, columns=columns)

    df.to_csv(join(eval_dir, "layers.csv"))

    plot_bar(dimensions, indexes, eval_dir)

    buf = BytesIO()
    dfi.export(df, buf)
    img = buf.getvalue()
    with open(join(eval_dir, 'dimensions.png'), 'wb') as file:
        file.write(img)



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


def plot_bar(dimensions, indexes, eval_dir):
    facecolor = 'w'
    color_red = 'r'
    color_blue = 'b'
    index = indexes
    column0 = dimensions[0]
    column1 = dimensions[1]
    title0 = 'Dense network structure'
    title1 = 'Sparse network structure'

    fig, axes = plt.subplots(figsize=(10, 5), facecolor=facecolor, ncols=2, sharey=True)
    fig.tight_layout()

    axes[0].barh(index, column0, align='center', color=color_red, zorder=10)
    axes[0].set_title(title0, fontsize=18, pad=15, color=color_red)
    axes[1].barh(index, column1, align='center', color=color_blue, zorder=10)
    axes[1].set_title(title1, fontsize=18, pad=15, color=color_blue)


    axes[0].invert_xaxis()
    plt.gca().invert_yaxis()

    axes[0].set(yticks=index, yticklabels=index)
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', colors='k')  # tick color

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    plt.savefig(join(eval_dir, "dimensions-bar.pdf"), facecolor=facecolor)

    return


def get_dimensions(args):
    layers = []
    model = gl.util.resume_model(args, 1)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layers.append(layer.weight.shape[0])
    return layers


def estimate_latency(args, repetitions=50):

    model = gl.util.resume_model(args, 1)
    device = torch.device("cuda")
    model.to(device)
    example_inputs = torch.randn(48, 3, 224, 224, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    for _ in range(10):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
    print(std_syn)
    return mean_syn, std_syn
