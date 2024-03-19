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

# maybe add more backend architectures
# TODO add a couple more architectures:
#  One or the other of the first two,
#  DenseNet121_Weights.IMAGENET1K_V1,
#  MobileNet_V3_Large_Weights.IMAGENET1K_V2,
#  EfficientNet_B3_Weights.IMAGENET1K_V1,
#  RegNet_Y_1_6GF_Weights.IMAGENET1K_V2,
#  ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1



def logs(args):
    start_time = datetime.now()
    args.save_dir = join(args.run_path, "logs",  start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")


args = parser.parse_arguments()

if __name__ == '__main__':

    if 't' in args.run_type:
        start_time = datetime.now()
        folderName = (f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{args.backbone}"
                      f"-{args.pruning_method}-{args.aggregation}")

        exists = True
        if not os.path.exists(join(args.run_path, args.backbone)):
            exists = False
            os.mkdir(join(args.run_path, args.backbone))

        args.run_path = join(args.run_path, folderName)
        run_path = args.run_path

        logs(args)

        s = join("0.0", "0.pth")
        save_path = join(run_path, s)

        os.mkdir(join(run_path, "0.0"))
        print(save_path)

        if args.backbone == "efficient":
            args.resize = [320, 300]

        model = wrap_train.wrap_train(args)
        torch.save(model, save_path)
        # will save the base trained network in a folder named as the backbone
        if not exists:
            torch.save(model, join(args.run_path, "..", args.backbone, "0.pth"))

    # prune network, produce list of networks at different all levels of sparsity
    if 'p' in args.run_type:
        if 't' not in args.run_type:
            if args.sparsity == 0.0:
                start_time = datetime.now()
                folderName = (f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}-{args.backbone}"
                              f"-{args.pruning_method}-{args.aggregation}")
                args.run_path = join(args.run_path, folderName)
                os.mkdir(args.run_path)
            logs(args)

        prune.prune(args)

    # evaluates all models within a directory
    if 'e' in args.run_type:
        if 't' not in args.run_type and 'p' not in args.run_type:
            logs(args)
        # TODO get rid of this
        vargs = vars(parser.parse_arguments())
        evaluate.evaluate(args, vargs)
