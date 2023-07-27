import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class fc_relu(nn.Module):

    def __init__(self, input_dims, out_dims, blk_type):
        super(fc_relu, self).__init__()
        if blk_type == "mlp_blk":
            self.fc = nn.Linear(input_dims, out_dims, bias=True)
        elif blk_type == "basic":
            self.fc = nn.Sequential(
                nn.Linear(input_dims, out_dims, bias=True),
                nn.Linear(input_dims, out_dims, bias=True),
            )
        elif blk_type == "ibn":
            self.fc = nn.Sequential(
                nn.Linear(input_dims, out_dims, bias=True),
                nn.Linear(input_dims, out_dims, bias=True),
                nn.Linear(input_dims, out_dims, bias=True),
            )
        self.act = nn.ReLU()
        self.pathid = -1
        self.blk_type = blk_type

    def forward(self, x, pathid=-1):
        self.pathid = pathid
        if self.blk_type == "mlp_blk":
            if pathid == -1:
                self.sigma_mean = self.dev(x, True)

                return self.act(self.fc(x)) + x
            elif pathid == 0:
                self.j = np.diag(np.ones(self.fc.in_features))
                self.sigma_mean = np.ones(self.fc.in_features)
                return x
            elif pathid == 1:
                self.j = self.jacobian(x)
                self.sigma_mean = self.dev(x)
                return self.act(self.fc(x))
        else:
            if pathid != 0:
                j = None
                input_x = x
                for layer in self.fc:
                    x = layer(x)
                    layer_j = self.jacobian(x, layer)
                    if j is None:
                        j = layer_j
                    else:
                        j = torch.matmul(j, layer_j)
                    x = self.act(x)
                # print(j.size())
                if pathid == 1:
                    self.sigma_mean = self.get_sigma(x, j, False)
                    return x

                elif pathid == -1:
                    self.sigma_mean = self.get_sigma(x, j, True)
                    return x + input_x
            elif pathid == 0:
                self.j = np.diag(np.ones(self.fc[0].in_features))
                self.sigma_mean = 1 
                return x

    def dev(self, x, skip=False):
        batch_num = x.size()[0]
        w = self.fc.weight.data
        dims = w.size()
        W = w.unsqueeze(0).repeat(batch_num, 1, 1)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        relu_dev = torch.where(x >= 0, one, zero)
        b = torch.zeros([batch_num, dims[1], dims[1]])
        for i in range(batch_num):
            for j in range(dims[1]):
                b[i, j, j] = relu_dev[i, j]
        j = torch.matmul(W, b)
        if skip:
            for i in range(batch_num):
                j[i] = j[i] + torch.from_numpy(np.identity(j[i].size()[1]))

        _, sigma, _ = torch.svd(j)
        # sigma=torch.mean(sigma,dim=(0,))
        # print(sigma.size())
        # exit()
        return sigma

    def jacobian(self, x, layer, skip=False):
        batch_num = x.size()[0]
        # print('batchnum', batch_num)
        w = layer.weight.data
        dims = w.size()
        W = w.unsqueeze(0).repeat(batch_num, 1, 1)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        relu_dev = torch.where(x >= 0, one, zero)
        b = torch.zeros([batch_num, dims[1], dims[1]])
        for i in range(batch_num):
            for j in range(dims[1]):
                b[i, j, j] = relu_dev[i, j]
        j = torch.matmul(W, b)
        return j

    def get_sigma(self, x, jac, skip):
        batch_num = x.size()[0]
        if skip:
            for i in range(batch_num):
                jac[i] = jac[i] + torch.from_numpy(np.identity(jac[i].size()[1]))

        _, sigma, _ = torch.svd(jac)
        return sigma


class ml_mlp(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, input_dims=784, num_classes=10, depth=10, width=10):
        super(ml_mlp, self).__init__()
        self.first_fc = nn.Linear(input_dims, width, bias=True)
        self.act = nn.ReLU()
        layer_list = []
        for i in range(depth):
            layer_list.append(fc_relu(width, width))
        self.block = nn.Sequential(*layer_list)
        self.last_fc = nn.Linear(width, num_classes, bias=True)
        self.depth = depth

    def forward(self, x, path_length=-1, source_id=0, target_id=10, feat_tag="ldi"):
        y = self.act(self.first_fc(x))
        if path_length <= -1 or path_length > self.depth:
            y = self.block(y)
        elif path_length == 0:
            if feat_tag == "di":
                self.j = 1
        else:
            path_id = np.random.choice(
                np.arange(self.depth), size=path_length, replace=False
            ).tolist()
            for blk_id, blk in enumerate(self.block):
                if blk_id in path_id:
                    y = blk(y, 1)
                else:
                    y = blk(y, 0)
            if feat_tag == "ldi":
                pass
            elif feat_tag == "di":
                j = 1
                for blkid, blk in enumerate(self.block):
                    if blkid == 0:
                        j = self.block[self.depth - 1 - blkid].j
                    else:
                        j = np.matmul(j, self.block[self.depth - 1 - blkid].j)
                self.j = j
        y = self.last_fc(y)
        if feat_tag == "ldi":
            pass
            all_sigma = []
            for blk in self.block:
                for i in blk.sigma_mean:
                    all_sigma.append(i)
            all_sigma = np.array(all_sigma)
            sig_mean = np.mean(all_sigma)
            sig_std = np.std(all_sigma)
            self.sig_mean = sig_mean
            self.sig_std = sig_std
        return y

    def extract_wgt(self, extract_first=False, extract_last=False):
        wgt_list = []
        if extract_first:
            wgt_list.append([self.first_fc.in_features, self.first_fc.out_features])
        for blk_id, blk in enumerate(self.block):
            wgt_list.append([blk.fc.in_features, blk.fc.out_features])
        if extract_last:
            wgt_list.append([self.last_fc.in_features, self.last_fc.out_features])
        return wgt_list

    def dev(self, fc_layer, layer_idx, x):
        batch_num = x.size()[0]
        if self.act == "relu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if self.act == "elu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.exp(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if layer_idx == 1:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            _, sigma, _ = torch.svd(W)
            return sigma
        if layer_idx == 0:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)

            if self.act == "relu":
                one = torch.ones_like(x)
                relu_dev = torch.where(x > 0, one, zero)
            if self.act == "elu":
                one = torch.exp(x)
                relu_dev = torch.where(x > 0, one, zero)

            b = torch.zeros([batch_num, dims[0], dims[0]])
            for i in range(batch_num):
                for j in range(dims[0]):
                    b[i, j, j] = relu_dev[i, j]

            j = torch.matmul(b, W)
            _, sigma, _ = torch.svd(j)
            return sigma

    def isometry(self, x):
        out0 = self.act_function(self.features[0](x))
        out1 = self.features[1](out0)

        in_dict = []
        in_dict.append(self.features[0](x))
        in_dict.append(out0)

        out_dict = []
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict = []
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1, out0), 1))
        for layer_idx in range(self.net_depth - 2):
            in_features = feat_dict[layer_idx]
            if self.layer_tc[layer_idx + 2] > 0:
                in_tmp = torch.cat(
                    (
                        out_dict[layer_idx + 1],
                        in_features[:, self.link_dict[layer_idx + 2]],
                    ),
                    1,
                )
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
            else:
                in_tmp = out_dict[layer_idx + 1]
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
        sigma_all = None
        for i in range(self.net_depth):
            sigma = self.dev(self.features[i], layer_idx=i, x=in_dict[i])
            if i == 0:
                sigma_all = sigma
            else:
                sigma_all = torch.cat((sigma_all, sigma), 1)

        sig_mean = sigma_all.view(-1).mean()
        sig_std = sigma_all.view(-1).std()
        return sig_mean, sig_std


class Dense_MLP(torch.nn.Module):
    def __init__(
        self, net_width, net_depth, tc, input_dims=2, num_classses=10, act_name="elu"
    ):
        super(Dense_MLP, self).__init__()
        self.analyze_topology(net_width, net_depth, tc)
        batchnorm_list = []
        layer_list = []
        layer_list.append(nn.Linear(input_dims, net_width + self.layer_tc[0]))
        batchnorm_list.append(nn.BatchNorm1d(net_width + self.layer_tc[0]))
        self.net_depth = net_depth
        self.act = act_name
        if act_name == "relu":
            self.act_function = F.relu
        if act_name == "elu":
            self.act_function = F.elu

        for i in range(net_depth - 2):
            layer_list.append(nn.Linear(net_width + self.layer_tc[i + 1], net_width))
            batchnorm_list.append(nn.BatchNorm1d(net_width + self.layer_tc[i + 1]))
        layer_list.append(
            nn.Linear(net_width + self.layer_tc[net_depth - 1], num_classses)
        )
        self.features = nn.ModuleList(layer_list).eval()
        self.batchnorm = nn.ModuleList(batchnorm_list).eval()
        self.link_dict = []
        for i in range(net_depth):
            self.link_dict.append(self.add_link(i))

        input = torch.randn(1, input_dims)
        self.params, self.flops = self.param_num()
        self.macs, self.params = profile(self, inputs=(input,))

    def param_num(self):
        num_param = 0
        flops = 0
        for layer in self.features:
            num_param = (
                num_param
                + (layer.in_features) * (layer.out_features)
                + layer.out_features
            )
            flops = (
                flops
                + 2 * (layer.in_features) * (layer.out_features)
                + layer.out_features
            )
        return num_param, flops

    def analyze_topology(self, net_width, net_depth, tc):
        all_path_num = np.zeros(net_depth)
        layer_tc = np.zeros(net_depth)
        for i in range(net_depth - 2):
            for j in range(i + 1):
                all_path_num[i + 2] = all_path_num[i + 2] + (net_width)
            layer_tc[i + 2] = min(tc, all_path_num[i + 2])

        self.layer_tc = np.array(layer_tc, dtype=int)
        self.all_path_num = np.array(all_path_num, dtype=int)
        self.density = (np.sum(layer_tc)) / (np.sum(all_path_num))
        self.nn_mass = self.density * net_width * net_depth

    def add_link(self, idx=0):
        tmp = list((np.arange(self.all_path_num[idx])))
        link_idx = random.sample(tmp, self.layer_tc[idx])
        link_params = torch.tensor(link_idx)
        return link_params

    def forward(self, x):
        out0 = self.act_function(self.features[0](x))
        out1 = self.features[1](out0)
        out_dict = []
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict = []
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1, out0), 1))
        for layer_idx in range(self.net_depth - 2):
            in_features = feat_dict[layer_idx]
            if self.layer_tc[layer_idx + 2] > 0:
                in_tmp = torch.cat(
                    (
                        out_dict[layer_idx + 1],
                        in_features[:, self.link_dict[layer_idx + 2]],
                    ),
                    1,
                )
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
            else:
                in_tmp = out_dict[layer_idx + 1]
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
        return out_dict[self.net_depth - 1]

    def dev(self, fc_layer, layer_idx, x):
        batch_num = x.size()[0]
        if self.act == "relu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if self.act == "elu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.exp(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if layer_idx == 1:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            _, sigma, _ = torch.svd(W)
            return sigma
        if layer_idx == 0:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)

            if self.act == "relu":
                one = torch.ones_like(x)
                relu_dev = torch.where(x > 0, one, zero)
            if self.act == "elu":
                one = torch.exp(x)
                relu_dev = torch.where(x > 0, one, zero)

            b = torch.zeros([batch_num, dims[0], dims[0]])
            for i in range(batch_num):
                for j in range(dims[0]):
                    b[i, j, j] = relu_dev[i, j]

            j = torch.matmul(b, W)
            _, sigma, _ = torch.svd(j)
            return sigma

    def isometry(self, x):
        out0 = self.act_function(self.features[0](x))
        out1 = self.features[1](out0)

        in_dict = []
        in_dict.append(self.features[0](x))
        in_dict.append(out0)

        out_dict = []
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict = []
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1, out0), 1))
        for layer_idx in range(self.net_depth - 2):
            in_features = feat_dict[layer_idx]
            if self.layer_tc[layer_idx + 2] > 0:
                in_tmp = torch.cat(
                    (
                        out_dict[layer_idx + 1],
                        in_features[:, self.link_dict[layer_idx + 2]],
                    ),
                    1,
                )
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
            else:
                in_tmp = out_dict[layer_idx + 1]
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
        sigma_all = None
        for i in range(self.net_depth):
            sigma = self.dev(self.features[i], layer_idx=i, x=in_dict[i])
            if i == 0:
                sigma_all = sigma
            else:
                sigma_all = torch.cat((sigma_all, sigma), 1)

        sig_mean = sigma_all.view(-1).mean()
        sig_std = sigma_all.view(-1).std()
        return sig_mean, sig_std


class prob_mlp(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self, input_dims=784, num_classes=10, depth=10, width=10, blk_type="mlp_blk"
    ):
        super(prob_mlp, self).__init__()
        self.first_fc = nn.Linear(input_dims, width, bias=True)
        self.act = nn.ReLU()
        layer_list = []
        for i in range(depth):
            layer_list.append(fc_relu(width, width, blk_type))
        self.block = nn.Sequential(*layer_list)
        self.last_fc = nn.Linear(width, num_classes, bias=True)
        self.depth = depth
        self.blk_type = blk_type

    def forward(
        self, x, path_length=-1, configs=None, source_id=0, target_id=10, feat_tag="ldi"
    ):
        y = self.act(self.first_fc(x))
        if configs is not None:
            self.net_config = ["input"]
            path_id = np.random.randint(0, high=2, size=len(self.block), dtype=int)
            for blk_id, blk in enumerate(self.block):
                if blk_id < len(configs) - 1:
                    if "skip" not in configs[blk_id + 1]:
                        y = blk(y, 1)
                        self.net_config.append(self.blk_type + "conv")
                    else:
                        y = blk(y, 0)
                        self.net_config.append(self.blk_type + "skip")
                else:
                    if path_id[blk_id] == 1:
                        y = blk(y, 1)
                        self.net_config.append(self.blk_type + "conv")
                    else:
                        y = blk(y, 0)
                        self.net_config.append(self.blk_type + "skip")
            self.net_config.append("output")
            if feat_tag == "ldi":
                pass
            elif feat_tag == "di":
                j = 1
                for blkid, blk in enumerate(self.block):
                    if blkid == 0:
                        j = self.block[self.depth - 1 - blkid].j
                    else:
                        # print(blkid, j, self.block[self.depth-1-blkid].j)
                        # print(blkid, j.size, self.block[self.depth-1-blkid].j.size)
                        j = np.matmul(j, self.block[self.depth - 1 - blkid].j)
                self.j = j
        elif path_length <= -1 or path_length > self.depth:
            y = self.block(y)
        elif path_length == 0:
            if feat_tag == "di":
                self.j = 1
        elif configs is None:
            self.net_config = ["input"]
            path_id = np.random.choice(
                np.arange(self.depth), size=path_length, replace=False
            ).tolist()
            for blk_id, blk in enumerate(self.block):
                if blk_id in path_id:
                    y = blk(y, 1)
                    self.net_config.append(self.blk_type + "conv")
                else:
                    y = blk(y, 0)
                    self.net_config.append(self.blk_type + "skip")
            self.net_config.append("output")
            if feat_tag == "ldi":
                pass

            elif feat_tag == "di":
                j = 1
                for blkid, blk in enumerate(self.block):
                    if blkid == 0:
                        j = self.block[self.depth - 1 - blkid].j
                    else:
                        # print(blkid, j, self.block[self.depth-1-blkid].j)
                        # print(blkid, j.size, self.block[self.depth-1-blkid].j.size)
                        j = np.matmul(j, self.block[self.depth - 1 - blkid].j)
                self.j = j
        y = self.last_fc(y)

        if feat_tag == "ldi":
            pass
            all_sigma = 0
            for blk in self.block:
                # for i in blk.sigma_mean:
                # print(blk.sigma_mean)
                all_sigma += blk.sigma_mean
            all_sigma = all_sigma / len(self.block)
            # print(all_sigma)
            all_sigma = np.array(all_sigma)
            # print(all_sigma)
            sig_mean = np.mean(all_sigma, axis=0)
            # print(sig_mean)
            # exit()
            sig_std = np.std(all_sigma, axis=0)
            # print(sig_mean,sig_std)
            self.sig_mean = all_sigma
            self.sig_std = sig_std
        return y

    def extract_wgt(self, extract_first=False, extract_last=False):
        wgt_list = []
        if extract_first:
            wgt_list.append([self.first_fc.in_features, self.first_fc.out_features])
        for blk_id, blk in enumerate(self.block):
            wgt_list.append([blk.fc.in_features, blk.fc.out_features])
        if extract_last:
            wgt_list.append([self.last_fc.in_features, self.last_fc.out_features])
        return wgt_list

    def dev(self, fc_layer, layer_idx, x):
        batch_num = x.size()[0]
        if self.act == "relu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if self.act == "elu" and layer_idx > 1:
            w = fc_layer.weight.data
            dims = w.size()
            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)
            one = torch.exp(x)
            relu_dev = torch.where(x > 0, one, zero)
            b = torch.zeros([batch_num, dims[1], dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i, j, j] = relu_dev[i, j]
            j = torch.matmul(W, b)
            _, sigma, _ = torch.svd(j)
            return sigma

        if layer_idx == 1:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            _, sigma, _ = torch.svd(W)
            return sigma
        if layer_idx == 0:
            w = fc_layer.weight.data
            dims = w.size()

            W = w.unsqueeze(0).repeat(batch_num, 1, 1)
            zero = torch.zeros_like(x)

            if self.act == "relu":
                one = torch.ones_like(x)
                relu_dev = torch.where(x > 0, one, zero)
            if self.act == "elu":
                one = torch.exp(x)
                relu_dev = torch.where(x > 0, one, zero)

            b = torch.zeros([batch_num, dims[0], dims[0]])
            for i in range(batch_num):
                for j in range(dims[0]):
                    b[i, j, j] = relu_dev[i, j]

            j = torch.matmul(b, W)
            _, sigma, _ = torch.svd(j)
            return sigma

    def isometry(self, x):
        out0 = self.act_function(self.features[0](x))
        out1 = self.features[1](out0)

        in_dict = []
        in_dict.append(self.features[0](x))
        in_dict.append(out0)

        out_dict = []
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict = []
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1, out0), 1))
        for layer_idx in range(self.net_depth - 2):
            in_features = feat_dict[layer_idx]
            if self.layer_tc[layer_idx + 2] > 0:
                in_tmp = torch.cat(
                    (
                        out_dict[layer_idx + 1],
                        in_features[:, self.link_dict[layer_idx + 2]],
                    ),
                    1,
                )
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
            else:
                in_tmp = out_dict[layer_idx + 1]
                in_dict.append(in_tmp)
                if layer_idx < self.net_depth - 3:
                    out_tmp = self.features[layer_idx + 2](self.act_function(in_tmp))
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp = self.features[layer_idx + 2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp, feat_dict[layer_idx + 1]), 1))
                    out_dict.append(out_tmp)
        sigma_all = None
        for i in range(self.net_depth):
            sigma = self.dev(self.features[i], layer_idx=i, x=in_dict[i])
            if i == 0:
                sigma_all = sigma
            else:
                sigma_all = torch.cat((sigma_all, sigma), 1)
        sig_mean = sigma_all.view(-1).mean()
        sig_std = sigma_all.view(-1).std()
        return sig_mean, sig_std
