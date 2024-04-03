import torch
from torch import nn
import torch.nn.functional as F

from .basic_module import (
    conv_nd,
    normalization,
    SiLU,
    linear,
    zero_module,
    timestep_embedding,
)
from .binary_cross_attention import CrossAttentionLayer


class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        use_conv=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(2, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            zero_module(
                conv_nd(2, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                2, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(2, channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class Channel_Mean_Pooling(nn.Module):
    def __init__(self):
        super(Channel_Mean_Pooling, self).__init__()
        self.mean_pooling = nn.AvgPool2d(
            kernel_size = (1, 2),
            stride = (1, 2)
        )
    def forward(self, x):
        x = x.transpose(1, 3) 
        x = self.mean_pooling(x)
        out = x.transpose(1, 3)
        return out


class LearnableBias(nn.Module):
    """
    time-dependent Bias
    """
    def __init__(self, emb_chn, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                out_chn,
            )),
        )

    def forward(self, x, emb):
        bias = self.bias(emb).unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = x + bias.expand_as(x)
        return out


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_chn))

    def forward(self, x):
        # real_weights = self.weight.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, bias=self.bias, stride=self.stride, padding=self.padding)

        return y


class BasicBlock(nn.Module):
    def __init__(self, emb_channels, inplanes, planes):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move1 = LearnableBias(emb_channels, inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, planes)
        self.bn = norm_layer(planes)

        self.Pooling = (inplanes != planes)
        if self.Pooling:
            self.pooling = Channel_Mean_Pooling()

        self.move2 = LearnableBias(emb_channels, planes)
        self.prelu = nn.PReLU(planes)
        self.move3 = LearnableBias(emb_channels, planes)
        self.binary_activation = BinaryActivation()

    def forward(self, x, emb):

        out = self.move1(x, emb)
        out = self.binary_activation(out)
        out = self.binary_3x3(out)
        out = self.bn(out)

        if self.Pooling:
            out += self.pooling(x)
        else:
            out += x

        out = self.move2(out, emb)
        out = self.prelu(out)
        out = self.move3(out, emb)
        return out


class DownBlock(nn.Module):
    def __init__(self, emb_channels, inplanes, planes):
        super(DownBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move1 = LearnableBias(emb_channels, inplanes)

        self.scalor = int(planes // inplanes)
        self.binary_3x3 = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        for i in range(self.scalor):
            self.binary_3x3.append(binaryconv3x3(inplanes, inplanes, stride=2))
            self.bn.append(norm_layer(inplanes))
        
        self.pooling = nn.AvgPool2d(2,2)

        self.move2 = LearnableBias(emb_channels, planes)
        self.prelu = nn.PReLU(planes)
        self.move3 = LearnableBias(emb_channels, planes)
        self.binary_activation = BinaryActivation()

    def forward(self, x, emb):

        out = self.move1(x, emb)
        out = self.binary_activation(out)
        out_list = []
        x = self.pooling(x)
        for i in range(self.scalor):
            out1 = self.binary_3x3[i](out)
            out1 = self.bn[i](out1)
            out1 += x
            out_list.append(out1)
        out = torch.cat(out_list, dim=1)

        out = self.move2(out, emb)
        out = self.prelu(out)
        out = self.move3(out, emb)
        return out


class UpBlock(nn.Module):
    def __init__(self, emb_channels, inplanes, planes):
        super(UpBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move1 = LearnableBias(emb_channels, inplanes)

        self.binary_3x3 = binaryconv3x3(inplanes, planes, stride=1)
        self.bn = norm_layer(planes)
        self.Pooling = (inplanes != planes)
        if self.Pooling:
            self.pooling = Channel_Mean_Pooling()
        self.move2 = LearnableBias(emb_channels, planes)
        self.prelu = nn.PReLU(planes)
        self.move3 = LearnableBias(emb_channels, planes)
        self.binary_activation = BinaryActivation()

    def forward(self, x, emb):
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        out = self.move1(x, emb)
        out = self.binary_activation(out)
        out = self.binary_3x3(out)
        out = self.bn(out)
        if self.Pooling:
            out += self.pooling(x)
        else:
            out += x

        out = self.move2(out, emb)
        out = self.prelu(out)
        out = self.move3(out, emb)
        return out

class ConcatLayer(nn.Module):
    def __init__(self, condition_dim, embedding_dim, emb_channels):
        super(ConcatLayer, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.condition_conv = ResBlock(condition_dim, emb_channels, embedding_dim)
        inplanes = 2 * embedding_dim
        planes = embedding_dim
        self.move1 = LearnableBias(emb_channels, inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, planes)
        self.bn = norm_layer(planes)

        self.Pooling = (inplanes != planes)
        if self.Pooling:
            self.pooling = Channel_Mean_Pooling()

        self.move2 = LearnableBias(emb_channels, planes)
        self.prelu = nn.PReLU(planes)
        self.move3 = LearnableBias(emb_channels, planes)
        self.binary_activation = BinaryActivation()

    def forward(self, c, x, emb):
        c = self.condition_conv(c, emb)
        x = torch.cat([x, c], dim=1)
        out = self.move1(x, emb)
        out = self.binary_activation(out)
        out = self.binary_3x3(out)
        out = self.bn(out)

        if self.Pooling:
            out += self.pooling(x)
        else:
            out += x

        out = self.move2(out, emb)
        out = self.prelu(out)
        out = self.move3(out, emb)
        return out

class UNetModel(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels,
        condition_dim_list,
        model_channels=128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, stride=1, padding=1)
        # first block with downsample
        self.block1_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels, planes=model_channels) # 256
        self.block1_down = DownBlock(emb_channels=time_embed_dim, inplanes=model_channels, planes=model_channels)
        
        # second block with downsample
        self.block2_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels, planes=model_channels) # 128
        self.block2_down = DownBlock(emb_channels=time_embed_dim, inplanes=model_channels, planes=model_channels * 2)

        # third block with downsample
        self.block3_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 2) # 64
        self.concat1 = ConcatLayer(condition_dim = condition_dim_list[0], embedding_dim=model_channels * 2, emb_channels=time_embed_dim)
        self.block3_down = DownBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 2)

        # forth block with downsample
        self.block4_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 2) # 32 
        self.concat2 = ConcatLayer(condition_dim = condition_dim_list[1], embedding_dim=model_channels * 2, emb_channels=time_embed_dim)
        self.block4_down = DownBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 4)

        # fifth block with downsample
        self.block5_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 4) # 16
        self.concat3 = ConcatLayer(condition_dim = condition_dim_list[2], embedding_dim=model_channels * 4, emb_channels=time_embed_dim)
        self.block5_down = DownBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 4)

        # middle block
        self.middle_block1 = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 4) # 8
        self.cross1 = CrossAttentionLayer(condition_dim = condition_dim_list[3], embedding_dim=model_channels * 4, emb_chn=time_embed_dim)
        self.middle_block2 = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 4) # 8

        # upsample block 1
        self.up_block1_up = UpBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 4)
        self.up_block1_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 8, planes=model_channels * 4) # 16

        # upsample block 2
        self.up_block2_up = UpBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 2)
        self.up_block2_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 2) # 32

        # upsample block 3
        self.up_block3_up = UpBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 2)
        self.up_block3_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 4, planes=model_channels * 2) # 64

        # upsample block 4
        self.up_block4_up = UpBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 1)
        self.up_block4_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 1) # 128

        # upsample block 5
        self.up_block5_up = UpBlock(emb_channels=time_embed_dim, inplanes=model_channels * 1, planes=model_channels * 1)
        self.up_block5_conv = BasicBlock(emb_channels=time_embed_dim, inplanes=model_channels * 2, planes=model_channels * 1) # 256

        self.out = nn.Sequential(
            normalization(model_channels),
            SiLU(),
            zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1)),
        )
    
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_conv.parameters()).dtype
    
    def forward(self, x, timesteps, feature_list, img, p=None):

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        x = torch.concat([img, x, p], dim=1)

        h = x.type(self.inner_dtype)

        h = self.input_conv(h)

        # first block
        h = self.block1_conv(h, emb)
        hs.append(h)  # 128, x1
        h = self.block1_down(h, emb)  # 128, x1

        # second block
        h = self.block2_conv(h, emb)
        hs.append(h)  # 128, x1
        h = self.block2_down(h, emb)  # 64, x2

        # third block
        h = self.block3_conv(h, emb)
        h = self.concat1(feature_list[0], h, emb)
        hs.append(h)  # 64, 2
        h = self.block3_down(h, emb)  # 32, x2

        # forth block
        h = self.block4_conv(h, emb)
        h = self.concat2(feature_list[1], h, emb)
        hs.append(h)  # 32, x2
        h = self.block4_down(h, emb)  # 16, x4

        # fifth block
        h = self.block5_conv(h, emb)
        h = self.concat3(feature_list[2], h, emb)
        hs.append(h)  # 16, x4
        h = self.block5_down(h, emb)  # 8, x4

        # middle block
        h = self.middle_block1(h, emb)
        h = self.cross1(feature_list[3], h, emb)
        h = self.middle_block2(h, emb)  # 8, x4

        # upsample block1
        h = self.up_block1_up(h, emb)
        h = self.up_block1_conv(torch.cat([h, hs[4]], dim=1), emb)  # 16, x4

        # upsample block2
        h = self.up_block2_up(h, emb)
        h = self.up_block2_conv(torch.cat([h, hs[3]], dim=1), emb)  # 32, x2

        # upsample block3
        h = self.up_block3_up(h, emb)
        h = self.up_block3_conv(torch.cat([h, hs[2]], dim=1), emb)  # 64, x2

        # upsample block4
        h = self.up_block4_up(h, emb)
        h = self.up_block4_conv(torch.cat([h, hs[1]], dim=1), emb)  # 128, x1

        # upsample block5
        h = self.up_block5_up(h, emb)
        h = self.up_block5_conv(torch.cat([h, hs[0]], dim=1), emb)  # 256, x1

        return torch.sigmoid(self.out(h))

    def convert_to_fp16(self):
        return None

    def convert_to_fp32(self):
        return None

