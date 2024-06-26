# should loop through all models in the test folder
# use the gl to get the recall @N for each model saved in text in each folder
# should then generate a graph of recall@N against sparsity from these
# graph should be saved in test folder

from os.path import join
import numpy as np
import pandas
import torchscan
from matplotlib import pyplot as plt
import os
import deep_visual_geo_localization_benchmark as gl
import torch
from io import BytesIO
import dataframe_image as dfi
import torch_pruning as tp
import format
def evaluate(args):


    size = round(args.max_sparsity/args.pruning_step)

    performance = np.zeros([size, 100])

    data = {"timings": np.zeros(size),
            "memory": np.zeros(size),
            "params": np.zeros(size),
            "macs": np.zeros(size),
            "dmas": np.zeros(size),
            "dimensions": np.zeros(size),
            "recall": np.zeros(size)}

    sparsity = np.zeros(size)

    dimensions = []
    columns = np.zeros(size)

    # lots of nesting!
    # is really just cuz each model will be in their own folder
    for f in os.listdir(args.run_path):
        dirpath = join(args.run_path, f)
        if os.path.isdir(dirpath):
            for g in os.listdir(dirpath):

                if os.path.isfile(join(dirpath, g)):

                    name = g.split(".")


                    if name[1] == "pth":
                        index = int(name[0])

                        sparsity[index] = (float("0." + name[0]))
                        args.resume = join(args.run_path, f, g)
                        args.save_dir = join(args.run_path, f)


                        model = gl.util.resume_model(args, 1)
                        device = torch.device("cuda")
                        model.to(device)
                        model.eval()

                        with torch.no_grad():

                            performance[index] = gl.eval(args)

                            example_inputs = torch.randn(512, 3, 224, 224, dtype=torch.float).to(device)

                            data["timings"][index] = (tp.utils.benchmark.measure_latency(model, example_inputs)[0])
                            data["memory"][index] = (tp.utils.benchmark.measure_memory(model, example_inputs, device))

                            module_info = torchscan.crawl_module(model, (3, 224, 224))

                            data["params"][index] = (module_info["overall"]["grad_params"] + module_info["overall"]["nograd_params"])
                            data["macs"][index] = (sum(layer["macs"] for layer in module_info["layers"]))
                            data["dmas"][index] = (sum(layer["dmas"] for layer in module_info["layers"]))


                        if len(dimensions) == 0:
                            dimensions = np.zeros([size, len(get_dimensions(args))])
                        dimensions[index] = (get_dimensions(args))
                        columns[index] = (float("0." + name[0]))

    performance = np.array(performance).T
    data["recall"] = performance[0]
    eval_dir = join(args.run_path, "eval")
    os.mkdir(eval_dir)

    for d in data:

        if d == "dimensions":
            continue

        percentages = [round((x / data[d][0]) * 100, 2) for x in data[d]]
        a = [data[d], percentages]
        np.savetxt(join(eval_dir, d + ".csv"), a, delimiter=",")

        plt.figure()
        plt.plot(sparsity, data[d], label=d)
        plt.legend()
        plt.xlabel("Sparsity")
        plt.xticks(sparsity)
        plt.ylabel(d)
        plt.title(f"{args.backbone}-{args.aggregation}-{args.pruning_method}")
        plt.savefig(join(eval_dir, d + ".pdf"))

    np.savetxt(join(eval_dir, "performance.csv"), performance, delimiter=",")
    np.savetxt(join(eval_dir, "dimensions.csv"), np.array(dimensions), delimiter=",")


    indexes = []
    for i in range(len(dimensions[0])):
        indexes.append(f"conv layer:{i}")

    df = pandas.DataFrame(data=np.array(dimensions).T, index=indexes, columns=columns)

    df.to_csv(join(eval_dir, "layers.csv"))

    buf = BytesIO()
    dfi.export(df, buf)
    img = buf.getvalue()
    with open(join(eval_dir, 'dimensions.png'), 'wb') as file:
        file.write(img)


    format.csv(eval_dir, f"{args.backbone}-{args.pruning_method}-{args.aggregation}")

def get_dimensions(args):
    layers = []
    model = gl.util.resume_model(args, 1)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layers.append(layer.weight.shape[0])
    return layers

