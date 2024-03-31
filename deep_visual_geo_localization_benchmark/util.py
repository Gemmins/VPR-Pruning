
import re

import dill
import torch
import shutil
import logging
import torchscan
import numpy as np
import torch_pruning as tp
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA

from deep_visual_geo_localization_benchmark import datasets_ws
from deep_visual_geo_localization_benchmark.model import network

def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    # TODO allow both with check
    #  but for now models aren't being loaded from state dicts
    model = network.GeoLocalizationNet(args).eval()
    state_dict = torch.load(args.resume, map_location=args.device)
    tp.load_state_dict(model, state_dict=state_dict)

    # model = torch.load(args.resume, map_location=args.device, pickle_module=dill)
    """
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    """
    return model


def resume_train(args, model=None):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    start_epoch_num = 0
    if model is None:
        model = network.GeoLocalizationNet(args).eval()
        state_dict = torch.load(args.resume, map_location=args.device)
        tp.load_state_dict(model, state_dict=state_dict)
        # model = torch.load(args.resume, pickle_module=dill)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_r5 = 0
    not_improved_num = 0
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")

    # need to test if this is  needed
    # model = torch.nn.DataParallel(model)

    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca
