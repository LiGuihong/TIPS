import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyn_slim.models.dyn_slim_ops import DSpwConv2d, DSdwConv2d, DSBatchNorm2d, DSBatchNorm2dSG, DSAvgPool2d, DSAdaptiveAvgPool2d, DSpwConv2dBN, DSConv2d
from timm.models.layers import sigmoid


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DSInvertedResidual(nn.Module):

    def __init__(self, in_channels_list, out_channels_list, kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSInvertedResidual, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [DSAvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False, channel_list=in_channels_list)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
        # Point-wise expansion
        self.conv_pw = DSpwConv2d(in_channels_list, mid_channels_list, bias=bias)
        self.bn1 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(mid_channels_list,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn2 = norm_layer(mid_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = None

        # Point-wise linear projection
        self.conv_pwl = DSpwConv2d(mid_channels_list, out_channels_list, bias=bias)
        self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.last_feature = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn3.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn3.set_zero_weight()

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def get_last_stage_distill_feature(self):
        return self.last_feature

    def forward(self, x):
        self._set_gate()

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual

        return x

    def _set_gate(self, set_pwl=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pwl:
            self.conv_pwl.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return 0

    def get_gate(self):
        return self.channel_choice


class DSInvertedResidualNoskipv2(nn.Module):

    def __init__(self, in_channels_list, out_channels_list, kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSInvertedResidualNoskipv2, self).__init__()

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None

        self.vital = in_channels_list[-1] != out_channels_list[-1] or stride == 2
        if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
            if len(self.out_channels_list)<2:
                pass
            else: 
                out_diff = self.out_channels_list[1]-self.out_channels_list[0]
                out_diff = int((out_diff+1)/2)
                out_len = len(self.out_channels_list)
                for i in range(len(self.out_channels_list)-1):
                    self.out_channels_list[out_len-i-2] = self.out_channels_list[out_len -i-1]-out_diff
            self.mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]

            if len(self.mid_channels_list)<2:
                pass
            else: 
                mid_diff = self.mid_channels_list[1]-self.mid_channels_list[0]
                mid_diff = int((mid_diff+1)/2)
                mid_len = len(self.mid_channels_list)
                for i in range(len(self.mid_channels_list)-1):
                    self.mid_channels_list[mid_len-i-2] = self.mid_channels_list[mid_len -i-1]-mid_diff

            self.conv_pw = DSpwConv2d(in_channels_list, mid_channels_list, bias=bias)
            self.bn1 = norm_layer(mid_channels_list, **norm_kwargs)
            self.act1 = act_layer(inplace=True)

            # Depth-wise convolution
            self.conv_dw = DSdwConv2d(mid_channels_list,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=bias)
            self.bn2 = norm_layer(mid_channels_list, **norm_kwargs)
            self.act2 = act_layer(inplace=True)

            # Channel attention and gating
            self.gate = MultiHeadGate(mid_channels_list,
                                    se_ratio=se_ratio,
                                    channel_gate_num=4 if has_gate else 0)

            # Point-wise linear projection
            self.conv_pwl = DSpwConv2d(mid_channels_list, out_channels_list, bias=bias)
            self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        else:
        # Point-wise expansion
            self.conv_pw = DSpwConv2d(in_channels_list, mid_channels_list, bias=bias)
            self.bn1 = norm_layer(mid_channels_list, **norm_kwargs)
            self.act1 = act_layer(inplace=True)

            # Depth-wise convolution
            self.conv_dw = DSdwConv2d(mid_channels_list,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=bias)
            self.bn2 = norm_layer(mid_channels_list, **norm_kwargs)
            self.act2 = act_layer(inplace=True)

            self.gate = None

            # Point-wise linear projection
            self.conv_pwl = DSpwConv2d(mid_channels_list, out_channels_list, bias=bias)
            self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        self.all_layer = [self.conv_pw,self.conv_dw,self.conv_pwl]
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.last_feature = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn3.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn3.set_zero_weight()

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def get_last_stage_distill_feature(self):
        return self.last_feature

    def forward(self, x):
        self._set_gate()

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if not(self.vital):
            x_channel= x.size()[1]
            res_channel = residual.size()[1]
            if x_channel>=res_channel:
                x[:,0:res_channel] = x[:,0:res_channel] + residual
            else:
                residual[:,0:x_channel] += residual[:,0:x_channel] + x
                x = residual
        return x

    def _set_gate(self, set_pwl=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pwl:
            self.conv_pwl.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return 0

    def get_gate(self):
        return self.channel_choice


class DSDepthwiseSeparable(nn.Module):

    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSDepthwiseSeparable, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None
        if self.has_residual:
            if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
                downsample_layers = []
                if stride == 2:
                    downsample_layers += [DSAvgPool2d(2, 2, ceil_mode=True,
                                                       count_include_pad=False, channel_list=in_channels_list)]
                if in_channels_list[-1] != out_channels_list[-1]:
                    downsample_layers += [DSpwConv2d(in_channels_list,
                                                     out_channels_list,
                                                     bias=bias)]
                self.downsample = nn.Sequential(*downsample_layers)
        # Depth-wise convolution
        self.conv_dw = DSdwConv2d(in_channels_list,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  bias=bias)
        self.bn1 = norm_layer(in_channels_list, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Channel attention and gating
        self.gate = None

        # Point-wise convolution
        self.conv_pw = DSpwConv2d(in_channels_list, out_channels_list, bias=bias)
        self.bn2 = norm_layer(out_channels_list, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn2.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn2.set_zero_weight()

    def forward(self, x):
        self._set_gate()
        residual = x

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Point-wise convolution
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.downsample is not None:
                residual = self.downsample(residual)
            if self.drop_path_rate > 0. and self.mode == 'largest':
                # Only apply drop_path on largest model
                x = drop_path(x, self.training, self.drop_path_rate)
            x += residual

        return x

    def _set_gate(self, set_pw=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pw:
            self.conv_pw.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return 0

    def get_gate(self):
        return self.channel_choice


class DSBasicBlockNoskipv2(nn.Module):
    
    def __init__(self, in_channels_list, out_channels_list, kernel_size, layer_num, stride=1, dilation=1,
                 act_layer=nn.ReLU, noskip=False, exp_ratio=6.0, se_ratio=0.25,
                 norm_layer=DSBatchNorm2d, norm_kwargs=None, conv_kwargs=None,
                 drop_path_rate=0., bias=False, has_gate=False):
        super(DSBasicBlockNoskipv2, self).__init__()

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.layer_num = layer_num
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]
        self.has_residual = not noskip
        self.drop_path_rate = drop_path_rate
        self.has_gate = has_gate
        self.downsample = None

        self.vital = in_channels_list[-1] != out_channels_list[-1] or stride == 2
        if in_channels_list[-1] != out_channels_list[-1] or stride == 2:
            if len(self.out_channels_list)<2:
                pass
            else: 
                out_diff = self.out_channels_list[1]-self.out_channels_list[0]
                out_diff = int((out_diff+1)/2)
                out_len = len(self.out_channels_list)
                for i in range(len(self.out_channels_list)-1):
                    self.out_channels_list[out_len-i-2] = self.out_channels_list[out_len -i-1]-out_diff
            self.mid_channels_list = [make_divisible(inc * exp_ratio) for inc in in_channels_list]

            if len(self.mid_channels_list)<2:
                pass
            else: 
                mid_diff = self.mid_channels_list[1]-self.mid_channels_list[0]
                mid_diff = int((mid_diff+1)/2)
                mid_len = len(self.mid_channels_list)
                for i in range(len(self.mid_channels_list)-1):
                    self.mid_channels_list[mid_len-i-2] = self.mid_channels_list[mid_len -i-1]-mid_diff

            self.conv_pw = DSConv2d(in_channels_list, out_channels_list, bias=bias, kernel_size=3, stride=stride)
            self.bn1 = norm_layer(out_channels_list, **norm_kwargs)
            self.act1 = act_layer(inplace=True)

            # Channel attention and gating
            self.gate = None

            # Point-wise linear projection
            self.conv_pwl = DSConv2d(out_channels_list, out_channels_list, bias=bias, kernel_size=3, stride=1)
            self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        else:
        # Point-wise expansion
            self.conv_pw = DSConv2d(in_channels_list, out_channels_list, bias=bias, kernel_size=3, stride=stride)
            self.bn1 = norm_layer(out_channels_list, **norm_kwargs)
            self.act1 = act_layer(inplace=True)

            # Channel attention and gating
            self.gate = None 

            # Point-wise linear projection
            self.conv_pwl = DSConv2d(out_channels_list, out_channels_list, bias=bias, kernel_size=3, stride=1)
            self.bn3 = norm_layer(out_channels_list, **norm_kwargs)
        self.channel_choice = -1
        self.mode = 'largest'
        self.next_channel_choice = None
        self.last_feature = None
        self.random_choice = 0
        self.init_residual_norm()

    def init_residual_norm(self, level='block'):
        if self.has_residual:
            if level == 'block':
                self.bn3.set_zero_weight()
            elif level == 'channel':
                self.bn1.set_zero_weight()
                self.bn3.set_zero_weight()

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def get_last_stage_distill_feature(self):
        return self.last_feature

    def forward(self, x):
        self._set_gate()

        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.has_gate:
            self.prev_channel_choice = self.channel_choice
            self.channel_choice = self._new_gate()
            self._set_gate(set_pwl=True)
        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if not(self.vital):
            x_channel= x.size()[1]
            res_channel = residual.size()[1]
            if x_channel>=res_channel:
                x[:,0:res_channel] = x[:,0:res_channel] + residual
            else:
                residual[:,0:x_channel] += residual[:,0:x_channel] + x
                x = residual
        return F.relu(x)

    def _set_gate(self, set_pwl=False):
        for n, m in self.named_modules():
            set_exist_attr(m, 'channel_choice', self.channel_choice)
        if set_pwl:
            self.conv_pwl.prev_channel_choice = self.prev_channel_choice
            if self.downsample is not None:
                for n, m in self.downsample.named_modules():
                    set_exist_attr(m, 'prev_channel_choice', self.prev_channel_choice)

    def _new_gate(self):
        if self.mode == 'largest':
            return -1
        elif self.mode == 'smallest':
            return 0
        elif self.mode == 'uniform':
            return self.random_choice
        elif self.mode == 'random':
            return random.randint(0, len(self.out_channels_list) - 1)
        elif self.mode == 'dynamic':
            return 0

    def get_gate(self):
        return self.channel_choice


def drop_path(inputs, training=False, drop_path_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_path_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)
