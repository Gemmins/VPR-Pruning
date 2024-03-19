import os
import torch
import argparse

# TODO add more options to backbone
# TODO look into aggregation options

def parse_arguments():
    parser = argparse.ArgumentParser(description="CNN pruning for VPR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # run
    parser.add_argument("--run_type", type=str, default=None,
                        help="Type of run to perform", choices=["tpe", "t", "tp", "te", "p", "pe", "e"])
    parser.add_argument("--run_path", type=str, default=None,
                        help="The path of the directory the run will be done in")

    # network
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backend of the network", choices=["resnet", "vgg", "resnet18conv5", "resnet18conv4", "dense",
                                                                "efficient", "mobile", "regnet", "shuffle", "resnet50conv5"])
    parser.add_argument("--aggregation", type=str, default=None,
                        help="Network aggregation layer", choices=["netvlad", "gem"])

    # training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")

    # dataset
    parser.add_argument("--dataset_train_name", type=str, default=None,
                        help="Name of dataset used in training")
    # this isn't used in the training loop, only for evaluating the performance of a network separately
    parser.add_argument("--dataset_eval_name", type=str, default=None,
                        help="Name of dataset used for evaluation")
    parser.add_argument("--datasets_folder", type=str, default=None,
                        help="path to datasets folder")

    # pruning
    parser.add_argument("--pruning_method", type=str, default=None,
                        help="Type of pruning to perform on the network", choices=["l1_norm", "l2_norm",
                                                                                   "hessian", "taylor", "bnScale",
                                                                                   "lamp", "random"])
    parser.add_argument("--max_sparsity", type=float, default=None,
                        help="The level of sparsity you want to go to")
    parser.add_argument("--pruning_step", type=float, default=None,
                        help="How much to prune each pass")
    parser.add_argument("--sparsity", type=float, default=0.0,
                        help="If network is already partially pruned, provide sparsity")
    parser.add_argument("--no_finetune", type=bool, default=False,
                        help="Fine-tune after each pruning step or not")


    parser.add_argument("--precompute", type=bool, default=False,
                        help="Flag to output computed descriptors and matching of model")

    # Paths parameters
    # TODO change to train_name later
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    # TODO change this later
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")

    ###################################################################################
    # all args below here are explicitly for deep-visual-geo-localization-benchmark

    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--lr_crn_layer", type=float, default=5e-3, help="Learning rate for the CRN layer")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4,
                        help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])


    # Model parameters
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None,
                        help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet",
                        choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))

    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop",
                                 "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01,
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")

    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=0, help="_")
    parser.add_argument("--contrast", type=float, default=0, help="_")
    parser.add_argument("--saturation", type=float, default=0, help="_")
    parser.add_argument("--hue", type=float, default=0, help="_")
    parser.add_argument("--rand_perspective", type=float, default=0, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0, help="_")
    parser.add_argument("--random_rotation", type=float, default=0, help="_")

    args = parser.parse_args()

    # TODO lots of these checks are unnecessary so eventually remove them

    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")

    if args.aggregation == "crn" and args.resume is None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")

    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")

    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError(
            "msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")

    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5",
                                 "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")

    if args.pca_dim is not None and args.pca_dataset_folder is None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")

    if args.backbone == "vit":
        if args.resize != [224, 224] and args.resize != [384, 384]:
            raise ValueError(f'Image size for ViT must be either 224 or 384 {args.resize}')
    if args.backbone == "cct384":
        if args.resize != [384, 384]:
            raise ValueError(f'Image size for CCT384 must be 384, but it is {args.resize}')

    if args.backbone in ["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                         "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5"]:
        if args.aggregation in ["cls", "seqpool"]:
            raise ValueError(f"CNNs like {args.backbone} can't work with aggregation {args.aggregation}")
    if args.backbone in ["cct384"]:
        if args.aggregation in ["spoc", "mac", "rmac", "crn", "rrm"]:
            raise ValueError(
                f"CCT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls, seqpool]")
    if args.backbone == "vit":
        if args.aggregation not in ["cls", "gem", "netvlad"]:
            raise ValueError(
                f"ViT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls]")

    ###################################################################################
    # all args below here are explicitly for VPR-bench

    parser.add_argument('-em', '--evalmode', required=True,
                        help='Specify Evaluation Mode (Possible value can be either of 0/1/2/3)', type=int)
    parser.add_argument('-sm', '--savematchinginfo', required=False, default=1,
                        help='Flag for storing matching data after computation (Possible value can be 0/1)', type=int)
    parser.add_argument('-dn', '--datasetname', default='Corridor', required=False,
                        help='Name of Dataset Used for Evaluation Mode 0. This is used for creating titles of plots.',
                        type=str)
    parser.add_argument('-ddir', '--datasetdirectory', default='datasets/corridor/', required=False,
                        help='Path to Dataset Directory Used for Evaluation Mode 0', type=str)
    parser.add_argument('-mdir', '--precomputedmatchesdirectory', default='precomputed_matches/corridor/',
                        required=False,
                        help='Optional Path to Precomputed Matches Directory Used for Evaluation Mode 0', type=str)
    parser.add_argument('-techs', '--VPRtechniquenames', nargs='+',
                        help='List of names of VPR techniques which could be any of these (NetVLAD,RegionVLAD,CoHOG,HOG,AlexNet_VPR,AMOSNet,HybridNet,CALC)',
                        required=True, type=str)

    return args
