import torch_pruning as tp
import torch
from os.path import join

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

# the method for pruning is given in args, should each have dummy method to allow for the addition of new methods


def prune(run_path, pruning_method, max_sparsity, pruning_step, current_sparsity):

    # TODO load model
    model_name = current_sparsity.split(".")[1] + ".pth"
    model_dir = join(run_path, current_sparsity, model_name)

    model = torch.load(model_dir)

    imp = get_imp(pruning_method)

    # TODO make sure this ends up an int
    iterative_steps = max_sparsity / pruning_step

    example_inputs = torch.randn(1, 3, 224, 224)

    pruner = tp.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=max_sparsity,
        global_pruning=True
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    for i in range(iterative_steps):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print(model(example_inputs).shape)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        print("=" * 16)

        # TODO finetune (train) here, then save the model



    return model


def get_imp(pruning_method):

    match pruning_method:
        case "l1_norm":
            imp = tp.importance.MagnitudeImportance(p=1, normalizer="mean", group_reduction="first")
        case "l2_norm":
            imp = tp.importance.MagnitudeImportance(p=2, normalizer="mean", group_reduction="first")
        case _:
            imp = None

    return imp