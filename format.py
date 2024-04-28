import os

import numpy as np
import pandas as pd

# This should go through a results folder and return a single dataframe
# Maybe also generate the latex for a table from the dataframe
# Sparsity along the top, fields should include
# Recall@1
# Area under recall@k curve
# Inference time (ms)
# Params (millions)
# DMAs (millions)
# MACs (millions)
# Peak memory usage (Gigabytes)




def csv(path: str, name: str) -> None:

    fields = ["Recall@1", "Recall@1 percentage",
              "Area under recall@k curve", "Area under recall@k curve percentage",
              "Inference time (ms)", "Inference time percentage",
              "Params (millions)", "Params percentage",
              "DMAs (millions)", "DMAs percentage",
              "MACs (millions)", "MACs percentage",
              "Peak memory usage (Gigabytes)", "Peak memory usage percentage"]


    sparsities = [round(x, 2) for x in np.linspace(0, 0.9, 10)]
    table = pd.DataFrame(index=sparsities, columns=fields)
    table.columns.name = "Sparsity Proportion"

    for f in os.listdir(path):
        l = f.split(".")
        if l[1] == "csv":
            match l[0]:
                case "recall":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x, 2) for x in data[0]]
                    table["Recall@1"] = d
                    table["Recall@1 percentage"] = data[1]
                case "performance":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    data = data.T
                    areas = []
                    percentage = []
                    for d in data:
                        area = round(np.trapz(d, dx=1)/100, 2)
                        areas.append(area)
                        percentage.append(round((area/areas[0])*100, 2))
                    table["Area under recall@k curve"] = areas
                    table["Area under recall@k curve percentage"] = percentage
                case "timings":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x, 2) for x in data[0]]
                    table["Inference time (ms)"] = d
                    table["Inference time percentage"] = data[1]
                case "dmas":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x / 1000000, 2) for x in data[0]]
                    table["DMAs (millions)"] = d
                    table["DMAs percentage"] = data[1]
                case "params":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x / 1000000, 2) for x in data[0]]
                    table["Params (millions)"] = d
                    table["Params percentage"] = data[1]
                case "macs":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x/1000000, 2) for x in data[0]]
                    table["MACs (millions)"] = d
                    table["MACs percentage"] = data[1]
                case "memory":
                    data = np.genfromtxt(os.path.join(path, f), delimiter=",")
                    d = [round(x / 1000000000, 2) for x in data[0]]
                    table["Peak memory usage (Gigabytes)"] = d
                    table["Peak memory usage percentage"] = data[1]

    table = table.T
    table.to_csv(os.path.join(path, name + ".csv"))
    table.to_latex(os.path.join(path, name + ".txt"))


