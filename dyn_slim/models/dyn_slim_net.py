import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dyn_slim.models.dyn_slim_blocks import DSInvertedResidual, DSDepthwiseSeparable, set_exist_attr, MultiHeadGate
from dyn_slim.models.dyn_slim_ops import DSConv2d, DSpwConv2d, DSBatchNorm2d, DSLinear, DSAdaptiveAvgPool2d
from dyn_slim.models.dyn_slim_stages import DSStage
from timm.models.layers import Swish
from timm.models.registry import register_model

__all__ = ['DSNet']

from dyn_slim.utils import efficientnet_init_weights


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


choices_cfgs = {  # outc, layer, kernel, stride, type, has_gate
    'slimmable_mbnet_v2_noskipv2_bn_uniform': [
        [[16, 48],                     1, 3, 2],
        [[14, 24],                     1, 3, 1, 'mbnv2nsv2', False],
        [[16, 32],                     2, 3, 2, 'mbnv2nsv2', False],
        [[18, 48],                     3, 3, 2, 'mbnv2nsv2', False],
        [[28, 88],                     4, 3, 2, 'mbnv2nsv2', False],
        [list(range(45, 136+1, 7)),    3, 3, 1, 'mbnv2nsv2', False],        
        [list(range(55, 224+1, 13)),   3, 3, 2, 'mbnv2nsv2', False],
        [list(range(97, 448 + 1, 27)), 1, 3, 1, 'mbnv2nsv2', False],
        [1280],  # has head
    ],
    # outc,                        layer, kernel, stride, type, has_gate, expansion_ratio, se_ratio, id_skip
    'slimmable_resnet34_noskipv2_bn_uniform': [
        [[24, 64],                     1, 7, 2],
        [[28, 64],                     3, 3, 2, 'rnnsv2', False],
        [[56, 128],                     4, 3, 2, 'rnnsv2', False],
        [list(range(100-39, 256+1, 12+3)),                     6, 3, 2, 'rnnsv2', False],
        [list(range(200-65, 512+1, 24+5)),                     3, 3, 2, 'rnnsv2', False],
        []
    ],
}



class DSNet(nn.Module):
    def __init__(self, num_classes=1000, in_chans=3,
                 choices_cfg=None, act_layer=nn.ReLU, noskip=False, drop_rate=0.,
                 drop_path_rate=0., se_ratio=0.25, norm_layer=DSBatchNorm2d,
                 norm_kwargs=None, bias=False, has_head=True, extra_sample_prob=0.0, **kwargs):
        super(DSNet, self).__init__()
        assert drop_path_rate == 0., 'drop connect not supported yet!'
        # logging.warning('Following args are not used when building DSNet:', kwargs)
        norm_kwargs = norm_kwargs or {}
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self._in_chs_list = [in_chans]

        # Stem
        stem_size, _, kernel_size, stride = choices_cfg[0]
        self.conv_stem = DSConv2d(self._in_chs_list,
                                  stem_size,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  bias=bias)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs_list = stem_size

        # Middle stages (IR/ER/DS Blocks)
        self.blocks = nn.ModuleList()
        for blkid, (out_channels_list, layer_num, kernel_size, stride, stage_type, has_gate) in enumerate(choices_cfg[1:-1]):
            if blkid == 0 and (stage_type in ['mbnv2','mbnv2ns','mbnv2nsv2'] or 'mbnv2' in stage_type):
                exp_ratio=1.0
            elif stage_type == 'mbnv2':
                exp_ratio=6.0
            else:
                exp_ratio=6.0
            self.blocks.append(DSStage(stage_type=stage_type,
                                       in_channels_list=self._in_chs_list,
                                       out_channels_list=out_channels_list,
                                       kernel_size=kernel_size,
                                       layer_num=layer_num,
                                       stride=stride,
                                       act_layer=act_layer,
                                       noskip=noskip,
                                       se_ratio=se_ratio,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs,
                                       drop_path_rate=drop_path_rate,
                                       bias=bias,
                                       exp_ratio=exp_ratio,
                                       has_gate=has_gate,
                                       extra_sample_prob=extra_sample_prob))
            self._in_chs_list = out_channels_list

        # Head + Pooling
        if has_head and len(choices_cfg[-1]) > 0:  # no head in mbnetv1
            self.has_head = True
            self.num_features = [choices_cfg[-1][0]]
            self.conv_head = DSpwConv2d(self._in_chs_list, self.num_features, bias=bias)
            self.bn2 = norm_layer(self.num_features, **norm_kwargs)
            self.act2 = act_layer(inplace=True)
        else:
            self.has_head = False
            self.num_features = self._in_chs_list
        self.global_pool = DSAdaptiveAvgPool2d(1, channel_list=self.num_features)

        # Classifier
        self.classifier = DSLinear(self.num_features, self.num_classes)

        with torch.no_grad():
            efficientnet_init_weights(self)
            self.init_residual_norm()
        self.set_mode('largest')
        self.head_channel_choice = None
        self.stem_channel_choice = None

    def set_mode(self, mode, seed=None, choice=None):
        self.mode = mode
        if seed is not None:
            random.seed(seed)
            seed += 1
        assert mode in ['largest', 'smallest', 'dynamic', 'uniform']
        for m in self.modules():
            set_exist_attr(m, 'mode', mode)
        if mode == 'largest':
            self.channel_choice = -1
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'smallest' or mode == 'dynamic':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
        elif mode == 'uniform':
            self.channel_choice = 0
            if self.has_head:
                self.set_module_choice(self.conv_head)
            self.channel_choice = 0
            if choice is not None:
                self.random_choice = choice
            else:
                self.random_choice = random.randint(1, 13)

        self.set_module_choice(self.conv_stem)
        self.set_module_choice(self.bn1)

    def set_uniform_mode(self, seed=None, choice=None):
        for idx, stage in enumerate(self.blocks):  # TODO: Optimize code
            if idx < 4:  # 3 for mbnet, 4 for effnet
                setattr(stage.first_block, 'channel_choice', 1)
                for m in stage.modules():
                    # print(m)
                    if hasattr(m, 'channel_choice'):
                        m.channel_choice=1
                        # print(m, m.channel_choice)
        setattr(self.conv_stem, 'channel_choice', 1)
        setattr(self.bn1, 'channel_choice', 1)
        # exit()
    
    def set_module_choice(self, m):
        set_exist_attr(m, 'channel_choice', self.channel_choice)

    def set_self_choice(self, m):
        self.channel_choice = m.channel_choice

    def init_residual_norm(self):
        for n, m in self.named_modules():
            if isinstance(m, DSInvertedResidual) or isinstance(m, DSDepthwiseSeparable):
                if m.has_residual:
                    logging.info('set block {} bn weight to zero'.format(n))
                m.init_residual_norm(level='block')

    def get_gate(self):
        gate = nn.ModuleList()
        for n, m in self.named_modules():
            if isinstance(m, MultiHeadGate) and m.has_gate:
                gate += [m.gate]
        return gate

    def get_bn(self):
        gate = nn.ModuleList()
        for n, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                gate += [m]
        print(gate)
        return gate

    def get_flops(self):
        net_flops = 0
        for m in self.modules():
            if hasattr(m, 'flops'):
                net_flops = net_flops + m.flops
        return net_flops

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool = DSAdaptiveAvgPool2d(1, channel_list=self.num_features)
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for idx, stage in enumerate(self.blocks):  # TODO: Optimize code
            if idx >= 4 and self.mode == 'uniform':  # 3 for mbnet, 4 for effnet
                setattr(stage.first_block, 'random_choice', self.random_choice)
            else:
                setattr(stage.first_block, 'random_choice', 0)
            self.set_module_choice(stage)
            if idx >= 0 and self.mode == 'uniform':  # 4 for mbnet, 5 for effnet
                setattr(stage, 'channel_choice', self.random_choice)
            x = stage(x)
            self.set_self_choice(stage)
        if self.has_head:
            self.set_module_choice(self.conv_head)
            self.set_module_choice(self.bn2)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
        self.set_module_choice(self.global_pool)
        self.set_module_choice(self.classifier)
        return x

    def forward_features_search(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for idx, stage in enumerate(self.blocks):  # TODO: Optimize code

            if idx >= 4 and self.mode == 'uniform':  # 3 for mbnet, 4 for effnet
                setattr(stage.first_block, 'random_choice', self.random_choice)
            else:
                setattr(stage.first_block, 'random_choice', 0)
            self.set_module_choice(stage)
            if idx >= 4 and self.mode == 'uniform':  # 4 for mbnet, 5 for effnet
                setattr(stage, 'channel_choice', self.random_choice)
            x = stage(x)
            self.set_self_choice(stage)
        if self.has_head:
            self.set_module_choice(self.conv_head)
            self.set_module_choice(self.bn2)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
        self.set_module_choice(self.global_pool)
        self.set_module_choice(self.classifier)
        return x
    
    def set_ratio(self, mode):
        if mode == 'teacher':
            for m in self.modules():
                if hasattr(m, 'prune_ratio'):
                    m.prune_ratio = 1
                    # print(m.prune_ratio)
            # exit()
            return 1
        elif mode == 'student':
            ratio_list=[]
            for stage in self.blocks:
                for m in stage.first_block.modules():
                        if hasattr(m, 'prune_ratio'):
                            m.prune_ratio = 1
                        ratio_list.append(1)
                for blk in stage.residual_blocks:
                    for m in blk.modules():
                        if hasattr(m, 'prune_ratio'):
                            upper_bound = m.out_channels_list[0]/m.out_channels_list[-1]
                            tmp_ratio = float(np.random.uniform(low=0.8,high=1.0001))
                            ratio_list.append(tmp_ratio)
                            m.prune_ratio = tmp_ratio+0.000000001                      
            return ratio_list

    def set_config(self, config):
        self.set_ratio( 'teacher')

        ratio_list=[]
        layer_id=0
        for stage in self.blocks:
            for m in stage.first_block.modules():
                    if hasattr(m, 'prune_ratio'):
                        m.prune_ratio = config[layer_id]+0.000000001
                    layer_id=layer_id+1
                    ratio_list.append(1)
            for blk in stage.residual_blocks:
                for m in blk.modules():
                    if hasattr(m, 'prune_ratio'):

                        m.prune_ratio = config[layer_id]+1e-8                     
                        layer_id=layer_id+1
      
    def forward(self, x, search=False, acc_pred = False, eval_config=False, blk_search=False, stage_idx=-1, blk_idx=-1, layer_idx=-1, low_bound=0.6,prune_ratio=0.6):
        x = self.forward_features(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0. and self.mode == 'largest':
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


@register_model
def slimmable_mbnet_v2_noskipv2_bn_uniform(pretrained=False, **kwargs):
    kwargs['noskip'] = True
    model = DSNet(
        choices_cfg=choices_cfgs['slimmable_mbnet_v2_noskipv2_bn_uniform'],
        pretrained=pretrained,
        act_layer=Swish,
        norm_layer=DSBatchNorm2d,
        **kwargs)
    return model


@register_model
def slimmable_resnet34_noskipv2_bn_uniform(pretrained=False, **kwargs):
    kwargs['noskip'] = True
    model = DSNet(
        choices_cfg=choices_cfgs['slimmable_resnet34_noskipv2_bn_uniform'],
        pretrained=pretrained,
        act_layer=Swish,
        norm_layer=DSBatchNorm2d,
        **kwargs)
    return model
