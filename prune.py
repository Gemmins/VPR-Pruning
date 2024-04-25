import copy
import logging
import os

import torch_pruning as tp
import torch
from os.path import join
import test
import wrap_train
import dill
from deep_visual_geo_localization_benchmark.eval import eval
from deep_visual_geo_localization_benchmark import datasets_ws
from deep_visual_geo_localization_benchmark.model.aggregation import NetVLAD
from deep_visual_geo_localization_benchmark.model.aggregation import GeM
from deep_visual_geo_localization_benchmark.model import network
# prune should take trained network + args and result in the
# creation of a number of pruned networks each in their own folder

# should prune from any sparsity in case of interruption of run
# will choose model to load based on current sparsity - does this by name

# pruning loop should:
#   prune model
#   train model
#   save model
#   increase current sparsity

# this should continue until max_sparsity has been reached
# at which point return

# model should be saved in run_path/<current_sparsity>/<current_sparsity>.pth
# the file name doesn't have the leading 0. i.e 0.8 -> 8 - just cuz I don't think '.' in file names is a good idea
# this naming is also assumed when loading a model
# therefore if loading a model not trained by this program then will need to rename it


def prune(args):
    sparsity = str(args.sparsity)
    # load model
    model_name = sparsity.split(".")[1] + ".pth"

    if not float(sparsity) == 0:
        model_dir = join(args.run_path, sparsity, model_name)
        model = network.GeoLocalizationNet(args).eval()
        state_dict = torch.load(model_dir, map_location=args.device, pickle_module=dill)
        tp.load_state_dict(model, state_dict=state_dict)
        args.resume = model_dir

    else:
        model_dir = join(args.run_path, "..", args.backbone, "0.pth")
        model = network.GeoLocalizationNet(args).eval()
        state_dict = torch.load(model_dir, map_location=args.device, pickle_module=dill)
        tp.load_state_dict(model, state_dict=state_dict)
        model = model.module
        save(sparsity, args, model)

    example_inputs = torch.randn(1, 3, 224, 224).to('cuda')

    model(example_inputs)

    model.zero_grad()
    sparsity = args.sparsity

    # get importance (this is effectively where the pruning method is chosen)
    imp = get_importance(args.pruning_method)

    iterative_steps = round((args.max_sparsity - args.sparsity) / args.pruning_step)

    example_inputs = torch.randn(1, 3, 224, 224).to('cuda')

    # this is just an odd way to stop the layer preceeding
    i = 0
    j = 0
    k = 0
    l = 0

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, NetVLAD):
            ignored_layers.append(m)
        elif isinstance(m, GeM):
            if args.backbone == "efficient":
                ignored_layers.append(i)
            else:
                ignored_layers.append(j)
        i = j
        j = k
        k = l
        l = m

    logging.info("Starting prune")


    pruner = tp.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=args.max_sparsity,
        global_pruning=True,
        ignored_layers=ignored_layers
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    #print(model(example_inputs).shape)

    for i in range(iterative_steps-1):
        pruner.step()

        # would be good to log/save this info somewhere
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        #print(model)
        #print(model(example_inputs).shape)
        logging.info(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps-1, base_nparams / 1e6, nparams / 1e6)
        )
        logging.info(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps-1, base_macs / 1e9, macs / 1e9)
        )
        logging.info("=" * 16)

        sparsity += args.pruning_step

        # this deals with floating point error, can increase if needed
        sparsity = round(sparsity, 5)

        base_macs, base_nparams = macs, nparams

        # obviously is dumb to save and reload but is simpler than doing anything else
        # save(sparsity, args, model)
        #recalls = eval(args, model)
        #print(*recalls)

        # finetune (train) here
        if not args.no_finetune:
            pruner.model = wrap_train.wrap_train(args, pruner=pruner, model=model)

        # save the fine-tuned model
        # recalls = eval(args, pruner.model)
        # print(*recalls)
        save(sparsity, args, pruner.model)


def get_importance(pruning_method):


    # choice here to be made about choosing first or mean group reduction
    # given that regardless the whole group will be removed
    # probs makes more sense to do mean but idk
    match pruning_method:

        case "random":
            # This doesn't work atm
            imp = tp.importance.RandomImportance
        case "l1_norm":
            imp = tp.importance.MagnitudeImportance(p=1, normalizer="mean", group_reduction="mean")
        case "l2_norm":
            imp = tp.importance.MagnitudeImportance(p=2, normalizer="mean", group_reduction="mean")
        case "fpgm":
            imp = tp.importance.FPGMImportance(p=2)
        case "lamp":
            imp = tp.importance.LAMPImportance(p=2)
        case _:
            imp = None

    return imp


def save(sparsity, args, model):
    # this is kinda wacky, might change naming at some point
    if not os.path.isdir(join(args.run_path, str(sparsity))):
        os.mkdir(join(args.run_path, str(sparsity)))
    model_dir = join(args.run_path, str(sparsity), str(sparsity).split(".")[1] + ".pth")
    state_dict = tp.state_dict(model)
    torch.save(state_dict, model_dir, pickle_module=dill)

    args.resume = model_dir
