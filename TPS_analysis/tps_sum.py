import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

import os, argparse
from model import prob_mlp

import math
import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Training MLP on MNIST/synthetic dataset")
parser.add_argument(
    "--batch_size", type=int, default=1000, help="Number of samples per mini-batch"
)
parser.add_argument("--epochs", type=int, default=1, help="Number of epoch to train")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--depth", type=int, default=3, help="the depth (number of FC layers) of the MLP"
)
parser.add_argument(
    "--width",
    type=int,
    default=80,
    help="the width (number of neurons per layers) of the MLP",
)
parser.add_argument(
    "--num_seg",
    type=int,
    default=2,
    help="the number of segmentation for the synthetic dataset",
)
parser.add_argument("--tc", type=int, default=20, help="the number of tc")
parser.add_argument("--dataset", type=str, default="MNIST", help="the type of dataset")
parser.add_argument(
    "--make_dataset",
    action="store_true",
    help="generate/regenerate the synthetic dataset or not.",
)
parser.add_argument(
    "--train_log_file",
    type=str,
    default="logs/mlp_train.logs",
    help="the name of file used to record the training/test record of MLPs",
)
parser.add_argument(
    "--res_log_file",
    type=str,
    default="logs/mlp_res.logs",
    help="the name of file used to record the training/test record of MLPs",
)
parser.add_argument(
    "--num_node",
    type=int,
    default=1,
    help="the number of iteration times to train the same architecture",
)

parser.add_argument(
    "--iter_times",
    type=int,
    default=1,
    help="the number of iteration times to train the same architecture",
)
args = parser.parse_args()


def generate_mlp(depth, blk_type="mlp_blk", sample_prob=0.4, skip_prob=0.5):
    net_config = ["input"]
    for i in range(depth):
        net_config.append(blk_type)
    net_config.append("output")
    adj_mat, special_index = generate_adj_matrix(
        net_config, sample_prob=sample_prob, skip_prob=skip_prob
    )
    stat_prob = page_rankv3(adj_mat, kersi=0.00001)

    return [stat_prob[0], np.max(stat_prob), np.min(stat_prob) * stat_prob[-1]]


def generate_mlp_by_config(net_config):
    adj_mat, special_index = generate_adj_matrix(net_config, sample_prob=-1)

    stat_prob = page_rankv3(adj_mat, kersi=0.00001)
    return stat_prob[1:-1]


def generate_mlp_by_depth(
    depth,
    blk_type="mlp_blk",
):
    net_config = ["input"]
    for i in range(depth):
        net_config.append(blk_type)
    net_config.append("output")
    adj_mat, special_index = generate_adj_matrix(net_config, sample_prob=-1)
    stat_prob = page_rankv3(adj_mat, kersi=0.00001)

    return stat_prob


def page_rankv3(adj_mat, kersi=0.00001):
    adj_mat = adj_mat / (np.sum(adj_mat, axis=0))

    tmp_adj_mat = adj_mat

    eg_num, eg_vec = np.linalg.eig(tmp_adj_mat)

    adj_mat = copy.deepcopy(tmp_adj_mat)
    for i in range(6000):
        adj_mat = np.matmul(adj_mat, tmp_adj_mat)

    stat_prob = adj_mat[:, 0]
    return stat_prob


def isReachable(s, d, adj_mat, num_nodes):
    visited = [False] * (num_nodes)

    queue = []

    queue.append(s)
    visited[s] = True

    while queue:
        # Dequeue a vertex from queue
        n = queue.pop(0)

        if n == d:
            return True

        for id, j in enumerate(adj_mat[n]):
            if j > 0:
                if visited[id] == False:
                    queue.append(id)
                    visited[id] = True

    return False


blk_dict = {
    "ibn": np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
    ),
    "input": np.array([[1, 1], [1, 0]]),
    "output": np.array([[0, 1], [1, 1]]),
    "ibnskip": np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    ),
    "ibnconv": np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    ),
    "bottleneck": np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
    ),
    "bottleneckskip": np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    ),
    "bottleneckconv": np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    ),
    "ibn_nores": np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    ),
    "basic": np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    ),
    "mlp_blk": np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    ),
    "mlp_skip": np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    ),
    "mlp_conv": np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    ),
    "basicskip": np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ),
    "basicconv": np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    ),
}


def generate_adj_matrix(
    net_config,
    prob_low=0,
    prob_high=1,
    num_step=10,
    interlayer="adj",
    sample_prob=0.2,
    skip_prob=0.5,
):
    num_nodes = 1
    sample_prob = np.random.uniform(prob_low, prob_high)
    for blk_config in net_config:
        num_nodes = num_nodes + blk_dict[blk_config].shape[0] - 1

    adj_mat = np.zeros((num_nodes, num_nodes))
    node_idx = 0
    special_index = {}
    for blk_config in net_config:
        special_index[blk_config] = node_idx

        blk_size = blk_dict[blk_config].shape[0]

        adj_mat[
            node_idx : node_idx + blk_size, node_idx : node_idx + blk_size
        ] = blk_dict[blk_config]
        node_idx = node_idx + blk_size - 1

    adj_mat = adj_mat[0:node_idx, 0:node_idx]
    num_link = np.count_nonzero(adj_mat)
    adj_mat[0, 0] = 1
    adj_mat[-1, -1] = 1
    return adj_mat, special_index


def generate_prob_adj_matrix(
    net_config,
    prob_low=0,
    prob_high=1,
    num_step=10,
    interlayer="adj",
    sample_prob=0.0,
    skip_prob=0.5,
):
    num_nodes = 1

    for blk_config in net_config:
        num_nodes = num_nodes + blk_dict[blk_config].shape[0] - 1

    adj_mat = np.zeros((num_nodes, num_nodes))
    node_idx = 0
    special_index = {}
    new_config = []
    for blk_config in net_config:
        if (
            blk_config not in ["input", "output", "mlp_blk"]
            and ("skip" not in blk_config)
            and ("conv" not in blk_config)
        ):
            prune_judge = np.random.uniform()

            if prune_judge < sample_prob:
                skip_judge = np.random.uniform()
                if skip_judge < skip_prob:
                    blk_config = blk_config + "skip"
                else:
                    blk_config = blk_config + "conv"
                special_index[blk_config] = node_idx
            else:
                special_index[blk_config] = node_idx

        blk_size = blk_dict[blk_config].shape[0]

        adj_mat[
            node_idx : node_idx + blk_size, node_idx : node_idx + blk_size
        ] = blk_dict[blk_config]
        node_idx = node_idx + blk_size - 1

    adj_mat = adj_mat[0:node_idx, 0:node_idx]
    num_link = np.count_nonzero(adj_mat)
    adj_mat[0, 0] = 1
    adj_mat[-1, -1] = 1

    return adj_mat, special_index


def generate_temporal_matrix(
    net_config,
    prob_low=0,
    prob_high=1,
    num_step=10,
    interlayer="adj",
    blk_config="basic",
    lambda_val=0.005,
    kersi=0.00001,
):
    all_adj_mat = None
    if isinstance(net_config, list):
        pass
    else:
        tmp_config = ["input"]
        for i in range(net_config):
            tmp_config.append(blk_config)
        tmp_config.append("output")
        net_config = tmp_config
    for i in range(num_step):
        adj_mat, tmp_id = generate_prob_adj_matrix(net_config)

        if i == 0:
            all_adj_mat = np.zeros(
                (num_step * adj_mat.shape[0], num_step * adj_mat.shape[0])
            )
        all_adj_mat[
            i * adj_mat.shape[0] : (i + 1) * adj_mat.shape[0],
            i * adj_mat.shape[0] : (i + 1) * adj_mat.shape[0],
        ] = adj_mat

    for i in range(num_step - 1):
        for j in range(i + 1, num_step):
            all_adj_mat[
                i * adj_mat.shape[0] : (i + 1) * adj_mat.shape[0],
                j * adj_mat.shape[0] : (j + 1) * adj_mat.shape[0],
            ] = np.identity(adj_mat.shape[0])
            all_adj_mat[
                j * adj_mat.shape[0] : (j + 1) * adj_mat.shape[0],
                i * adj_mat.shape[0] : (i + 1) * adj_mat.shape[0],
            ] = np.identity(adj_mat.shape[0])

    all_adj_mat = all_adj_mat  # *(1-kersi)+kersi#*np.identity(all_adj_mat.shape[0])

    all_stat_prob = page_rankv3(all_adj_mat)
    stat_prob = np.zeros(adj_mat.shape[0])
    for i in range(num_step):
        stat_prob += all_stat_prob[i * adj_mat.shape[0] : (i + 1) * adj_mat.shape[0]]

    return stat_prob  # /stat_prob.shape[0]


def mnist_dataloaders(
    train_batch_size=64,
    test_batch_size=100,
    num_workers=2,
    cutout_length=4,
    data_dir="./data",
):
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader


def train(
    images,
    labels,
    optimizer,
    model,
    criterion,
    device="cpu",
    path_length=-1,
    net_configs=None,
):
    input_size = images.size()
    images = (
        Variable(images).view(input_size[0], input_size[2] * input_size[2]).to(device)
    )
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    output = model(images, path_length=path_length, configs=net_configs)

    return model.sig_mean, model.net_config


def calculate_tps(sta_prob, total_depth, path_length):
    tps = sta_prob[0] * sta_prob[1] * sta_prob[0] * sta_prob[1]
    for i in range(total_depth - path_length):
        tps = tps * sta_prob[1]
    for i in range(path_length):
        tps = tps * sta_prob[2]
    return tps


def calculate_tps_by_path(sta_prob):
    tps = 1
    for i in sta_prob:
        tps = tps * i
    return tps


def calculate_tps_by_depth(
    sta_prob,
    net_config,
    model,
    blk_type="basic",
):
    tps = 0
    node_id = 0
    length = 0

    blk_id = 0
    sigma_sum = 0
    print(net_config)
    for blk_config in net_config:
        if blk_config in ["input", "output"]:
            node_id += blk_dict[blk_config].shape[0] - 1
            pass
        else:
            if "skip" in blk_config:
                tps = tps + sta_prob[node_id]
                length = length + 1

            elif "conv" in blk_config:
                length = length + blk_dict[blk_type].shape[0] - 1
                for j in range(blk_dict[blk_type].shape[0] - 1):
                    tps = tps + sta_prob[node_id + j]

            node_id += blk_dict[blk_type].shape[0] - 1
            sigma_sum = sigma_sum + model.block[blk_id].sigma_mean

            blk_id = blk_id + 1

    all_sigma = sigma_sum / length
    all_sigma = np.array(all_sigma)

    sig_mean = np.mean(all_sigma, axis=(0, 1))

    sig_std = np.std(all_sigma, axis=(0, 1))
    print(length, tps, sig_mean, sig_std)
    return tps, sig_mean, sig_std


def solve_equation(num_node, blk_type, depth):
    num_skip_node = 1
    num_blk_node = blk_dict[blk_type].shape[0] - 1
    residual = int(num_node % num_blk_node)
    solution = []
    for skip_num in range(residual, depth, num_blk_node):
        if (
            skip_num + int((num_node - skip_num) / num_blk_node) <= depth
            and int((num_node - skip_num) / num_blk_node) > 0
        ):
            solution.append([skip_num, int((num_node - skip_num) / num_blk_node)])
    print("solution\n", solution)

    net_config = []
    for sol in solution:
        skip_num, blk_num = sol[0], sol[1]
        single_config = ["input"]
        skipid = np.random.choice(
            np.arange(sol[0] + sol[1]), size=skip_num, replace=False
        ).tolist()
        for i in range(sol[0] + sol[1]):
            if i < skip_num:
                single_config.append(blk_type + "skip")
            else:
                single_config.append(blk_type + "conv")
        net_config.append(single_config)
    print("net_config\n", net_config)
    return net_config


def main(args):
    train_loader, test_loader = mnist_dataloaders(args.batch_size)
    tps_dict = {}
    ldi_dict = {}

    depth = args.depth
    args.depth = depth
    model = prob_mlp(
        width=args.width,
        depth=args.depth,
        input_dims=784,
        num_classes=10,
        blk_type="basic",
    )

    stat_prob = generate_temporal_matrix(args.depth, blk_config="basic", num_step=1)
    print("stationary distribution")
    print(stat_prob)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_acc = -1000

    tps_dict[depth] = []
    ldi_dict[depth] = []

    data = []
    for i, (images, labels) in enumerate(train_loader):
        if i > 0:
            break
        node_config = solve_equation(args.num_node, blk_type="basic", depth=args.depth)
        model = prob_mlp(
            width=args.width,
            depth=args.depth,
            input_dims=784,
            num_classes=10,
            blk_type="basic",
        )
        for configs in node_config:
            grad_mg, net_config = train(
                images,
                labels,
                optimizer=optimizer,
                model=model,
                criterion=criterion,
                path_length=-10,
                net_configs=configs,
            )
            grad_mg = np.mean(grad_mg, axis=(1,))
            tps, sig_mean, sig_std = calculate_tps_by_depth(stat_prob, configs, model)
            data.append([tps, sig_mean, sig_std])
        del model

    plt.rcParams.update({"font.size": 20})
    plt.rcParams["legend.loc"] = "upper left"
    tps_array = np.array(data)
    np.savetxt("sim_{}_{}_tas_loop_sum.csv".format(depth, args.num_node), tps_array)

    font_size = 15
    legend = []
    plt.plot(tps_array[:, 0], tps_array[:, 1], "-s")
    plt.tight_layout()
    plt.grid(linestyle=":")

    plt.xlabel("TPS (log scale)", fontsize=font_size)
    plt.ylabel("Mean Singular Value (log scale)", fontsize=font_size)
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=font_size - 2)
    plt.savefig("figs/sim_{}_{}_jst_tsa_sum.png".format(args.width, args.depth))
    plt.savefig("figs/sim_{}_{}_jst_tsa_sum.svg".format(args.width, args.depth))
    plt.savefig("figs/sim_{}_{}_jst_tsa_sum.pdf".format(args.width, args.depth))
    plt.close()


if __name__ == "__main__":
    main(args)
