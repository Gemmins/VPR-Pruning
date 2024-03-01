import os
import parser
import wrap_train
import prune
import evaluate
import logging
import torch
from os.path import join
from datetime import datetime
from deep_visual_geo_localization_benchmark import commons

# 1. Train networks from scratch on a dataset
# 2. Perform pruning and fine-tuning of a trained network
# 3. Evaluate performance metrics of a network and generate relevant figures

# Should be able to do all three in one command for any specified network architecture
# Eventually be able to do any combination of the three in order

# Should allow easy addition of custom training/pruning
# Currently will only work with torch-prune style methods

# Do I want to have the pruning method loop or should I loop within it?
# Currently, the looping is done within the pruning method, for torch-prune
# methods this makes sense

# Output of a full experiment maybe should be something like this:
# .
# └── <run_path>
#     └── <run_dir>
#         ├── logs
#         ├── 0.0
#         │   ├── 0.pth
#         │   ├── eval
#         │   └── 0_logging.txt
#         ├── 0.1
#         │   ├── 1.pth
#         │   ├── eval
#         │   └── 1_logging.txt
#         │
#         recall_graph.pdf
#         │
#         etc

# TODO add more backend architectures
# TODO maybe add more aggregations
# TODO logging


def logs(args):
    start_time = datetime.now()
    args.save_dir = join("logs", args.run_path, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")


args = parser.parse_arguments()

if __name__ == '__main__':

    if 't' in args.run_type:
        start_time = datetime.now()
        folderName = (f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{args.backbone}-"
                      f"{args.aggregation}-{args.pruning_method}")
        args.run_path = join(args.run_path, folderName)
        run_path = args.run_path

        logs(args)

        s = join("0.0", "0.pth")
        save_path = join(run_path, s)

        os.mkdir(run_path)
        os.mkdir(join(run_path, "0.0"))
        print(save_path)

        torch.save(wrap_train.wrap_train(args), save_path)

    # prune network, produce list of networks at different all levels of sparsity
    if 'p' in args.run_type:
        if 't' not in args.run_type:
            args = parser.parse_arguments()

        prune.prune(args)

    # evaluates all models within a directory
    if 'e' in args.run_type:
        if 't' not in args.run_type and 'p' not in args.run_type:
            args = parser.parse_arguments()
        # TODO get rid of this
        vargs = vars(parser.parse_arguments())
        evaluate.evaluate(args, vargs)
