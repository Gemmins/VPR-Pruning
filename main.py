import os

import parser
import wrap_train
import prune
import evaluate
import logging
import torch
from os.path import join

# 1. Train networks from scratch on a dataset
# 2. Perform pruning and fine-tuning of a trained network
# 3. Evaluate performance metrics of a network and generate relevant figures

# Should be able to do all three in one command for any specified network architecture
# Able to do any combination of the three in order

# Should allow easy addition of custom training/pruning
# Currently will only work with torch-prune style methods

# Do I want to have the pruning method loop or should I loop within it?
# Currently, the looping is done within the pruning method, for torch-prune
# methods this makes sense

# Output of a full experiment maybe should be something like this:
# .
# └── <run_path>
#     ├── logging.txt
#     ├── 0.0
#     │   ├── 0.pth
#     │   ├── computed
#     │   └── 0_logging.txt
#     ├── 0.1
#     │   ├── 1.pth
#     │   ├── eval
#     │   └── 1_logging.txt
#    etc

# TODO add more pruning methods
# TODO add more backend architectures
# TODO maybe add more aggregations
# TODO implement evaluation
# TODO logging

args = parser.parse_arguments()
run_path = args.run_path
if __name__ == '__main__':

    if 't' in args.run_type:

        s = join("0.0", "0.pth")
        save_path = join(run_path, s)

        os.mkdir(run_path)
        os.mkdir(join(run_path, "0.0"))
        print(save_path)

        torch.save(wrap_train.wrap_train(args), save_path)

    # prune network, produce list of networks at different all levels of sparsity
    if 'p' in args.run_type:
        prune.prune(args)

    # evaluates all models within a directory
    if 'e' in args.run_type:
        args = vars(parser.parse_arguments())
        evaluate.evaluate(run_path, args)
