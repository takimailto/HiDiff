import torch
from torch import nn
import torch.nn.functional as F

import math
from .basic_module import (
    conv_nd,
    normalization,
    SiLU,
    linear,
    zero_module,
)


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


class BatchNormT(nn.BatchNorm1d):
    def __init__(self, *kargs, eps=1e-5):
        super(BatchNormT, self).__init__(*kargs, eps=eps)
    
    def forward(self, input):
        # Need to transpose unlike original nn.BatchNorm1d
        input = input.transpose(-1, -2)

        self._check_input_dim(input)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Output needs to be de-transposed unlike original nn.BatchNorm1d
        return nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        ).transpose(-1, -2)


class RPReLU(nn.Module):
    def __init__(self, emb_chn, hidden_size):
        super().__init__()
        self.move1 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                hidden_size,
            )),
        )
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                hidden_size,
            )),
        )

    def forward(self, x, emb):
        out = self.prelu((x + self.move1(emb).unsqueeze(dim=1)).transpose(-1, -2)).transpose(-1, -2) + self.move2(emb).unsqueeze(dim=1)
        return out


class LayerScale(nn.Module):
    def __init__(self, emb_chn, hidden_size):
        super().__init__()
        self.alpha = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                hidden_size,
            )),
        )
        self.move = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                hidden_size,
            )),
        )

    def forward(self, x, emb):
        out = x * self.alpha(emb).unsqueeze(dim=1) + self.move(emb).unsqueeze(dim=1)
        return out


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim, emb_chn, reduction_ratio=1):
        super().__init__()

        self.reduction_ratio = reduction_ratio
        self.embedding_dim = embedding_dim

        self.num_attention_heads = 4
        self.attention_head_size = embedding_dim // self.num_attention_heads

        self.moveq1 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.movek1 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.movev1 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )

        self.query1 = nn.Linear(embedding_dim, embedding_dim)
        self.key1 = nn.Linear(embedding_dim, embedding_dim)
        self.value1 = nn.Linear(embedding_dim, embedding_dim)

        self.normq1 = BatchNormT(embedding_dim, eps=1e-5)
        self.normk1 = BatchNormT(embedding_dim, eps=1e-5)
        self.normv1 = BatchNormT(embedding_dim, eps=1e-5)

        if self.reduction_ratio > 1:
            self.pool = nn.AvgPool2d(self.reduction_ratio)
            self.mover = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
            self.reduction = nn.Linear(embedding_dim, embedding_dim)
            self.norm_r = BatchNormT(embedding_dim, eps=1e-5)
            self.rprelur = RPReLU(emb_chn, embedding_dim)
        
        self.rpreluq1 = RPReLU(emb_chn, embedding_dim)
        self.rpreluk1 = RPReLU(emb_chn, embedding_dim)
        self.rpreluv1 = RPReLU(emb_chn, embedding_dim)

        self.moveq2 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.movek2 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.movev2 = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )

        self.norm_context = BatchNormT(embedding_dim, eps=1e-5)
        self.rprelu_context = RPReLU(emb_chn, embedding_dim)

    def forward(self, condition_embedding, embedding_embedding, emb):

        mixed_query_layer = self.normq1(self.query1(embedding_embedding + self.moveq1(emb).unsqueeze(dim=1))) + embedding_embedding

        if self.reduction_ratio > 1:
            B, N, C = condition_embedding.shape
            image_height = int(math.sqrt(N))
            image_width = int(math.sqrt(N))
            condition_embedding = condition_embedding.permute(0, 2, 1).reshape(B, C, image_height, image_width)
            condition_embedding = self.pool(condition_embedding).reshape(B, C, -1).permute(0, 2, 1)
            condition_embedding = self.norm_r(self.reduction(condition_embedding + self.mover(emb).unsqueeze(dim=1))) + condition_embedding
            condition_embedding = self.rprelur(condition_embedding, emb)

        mixed_key_layer = self.normk1(self.key1(condition_embedding + self.movek1(emb).unsqueeze(dim=1))) + condition_embedding
        mixed_value_layer = self.normv1(self.value1(condition_embedding + self.movev1(emb).unsqueeze(dim=1))) + condition_embedding

        mixed_query_layer = self.rpreluq1(mixed_query_layer, emb)
        mixed_key_layer = self.rpreluk1(mixed_key_layer, emb)
        mixed_value_layer = self.rpreluv1(mixed_value_layer, emb)

        query_layer = mixed_query_layer + self.moveq2(emb).unsqueeze(dim=1)
        key_layer = mixed_key_layer + self.movek2(emb).unsqueeze(dim=1)
        value_layer = mixed_value_layer + self.movev2(emb).unsqueeze(dim=1)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embedding_dim,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.reduction_ratio > 1:
            mixed_key_layer = mixed_key_layer.permute(0, 2, 1).reshape(B, C, image_height // self.reduction_ratio, 
                                                                       image_width // self.reduction_ratio)
            mixed_key_layer = torch.nn.functional.interpolate(mixed_key_layer, size=image_height, mode='nearest').reshape(B, C, -1).permute(0, 2, 1)

            mixed_value_layer = mixed_value_layer.permute(0, 2, 1).reshape(B, C, image_height // self.reduction_ratio, 
                                                                           image_width // self.reduction_ratio)
            mixed_value_layer = torch.nn.functional.interpolate(mixed_value_layer, size=image_height, mode='nearest').reshape(B, C, -1).permute(0, 2, 1)

        context_layer = self.norm_context(context_layer) + mixed_query_layer + mixed_key_layer + mixed_value_layer
        context_layer = self.rprelu_context(context_layer, emb)

        return context_layer
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # B d N C/d
        
        
class LayerScaleModule(nn.Module):

    def __init__(self, embedding_dim, emb_chn):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)

        self.move = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.norm = BatchNormT(embedding_dim, eps=1e-5)
        self.rprelu = RPReLU(emb_chn, embedding_dim)

        self.layerscale = LayerScale(emb_chn, embedding_dim)

    def forward(self, hidden_states, emb):

        out = self.norm(self.dense(hidden_states + self.move(emb).unsqueeze(dim=1))) + hidden_states
        out = self.rprelu(out, emb)

        out = self.layerscale(out, emb)

        return out


class ViTIntermediate(nn.Module):
    def __init__(self, embedding_dim, emb_chn):
        super().__init__()
        self.expansion_ratio = 4
        self.dense = nn.Linear(embedding_dim, embedding_dim * self.expansion_ratio)

        self.move = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim,
            )),
        )
        self.norm = BatchNormT(embedding_dim * self.expansion_ratio, eps=1e-5)
        self.rprelu = RPReLU(emb_chn, embedding_dim * self.expansion_ratio)


    def forward(self, hidden_states, emb):

        out = self.norm(self.dense(hidden_states + self.move(emb).unsqueeze(dim=1))) + torch.concat([hidden_states for _ in range(self.expansion_ratio)], dim=-1)
        out = self.rprelu(out, emb)

        return out



class ViTOutput(nn.Module):
    def __init__(self, embedding_dim, emb_chn):
        super().__init__()
        self.expansion_ratio = 4
        self.dense = nn.Linear(embedding_dim * self.expansion_ratio, embedding_dim)

        self.move = nn.Sequential(
            SiLU(),
            zero_module(linear(
                emb_chn,
                embedding_dim * self.expansion_ratio,
            )),
        )
        self.norm = BatchNormT(embedding_dim, eps=1e-5)
        self.rprelu = RPReLU(emb_chn, embedding_dim)
        self.pooling = nn.AvgPool1d(self.expansion_ratio)
        self.layerscale = LayerScale(emb_chn, embedding_dim)

    def forward(self, hidden_states, emb):
        out = self.norm(self.dense(hidden_states + self.move(emb).unsqueeze(dim=1))) + self.pooling(hidden_states)
        out = self.rprelu(out, emb)

        out = self.layerscale(out, emb)

        return out


class FNN(nn.Module):
    def __init__(self, embedding_dim, emb_chn):
        super().__init__()
        self.layernorm_before = BatchNormT(embedding_dim, eps=1e-5)
        self.layernorm_after = BatchNormT(embedding_dim, eps=1e-5)
        self.intermediate = ViTIntermediate(embedding_dim, emb_chn)
        self.output = ViTOutput(embedding_dim, emb_chn)

        kernel_size = 5
        self.avg_res_w5 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
        self.layerscale_w5 = LayerScale(emb_chn, embedding_dim)
        self.avg_res_h5 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
        self.layerscale_h5 = LayerScale(emb_chn, embedding_dim)

        kernel_size = 3
        self.avg_res_w3 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
        self.layerscale_w3 = LayerScale(emb_chn, embedding_dim)
        self.avg_res_h3 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
        self.layerscale_h3 = LayerScale(emb_chn, embedding_dim)

    def forward(self, hidden_states, attention_output, emb):
        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        hidden_states_norm = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(hidden_states_norm, emb)

        # second residual connection is done here
        layer_output = self.output(layer_output, emb) + hidden_states
        B, N, C = hidden_states_norm.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))

        layer_output += self.layerscale_h3(self.avg_res_h3(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous(), emb)
        layer_output += self.layerscale_w3(self.avg_res_w3(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous(), emb)
        
        layer_output += self.layerscale_h5(self.avg_res_h5(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous(), emb)
        layer_output += self.layerscale_w5(self.avg_res_w5(hidden_states_norm.permute(0, 2, 1).view(-1, C, H, W).contiguous()).view(-1, C, N).permute(0, 2, 1).contiguous(), emb)

        return layer_output

class CrossAttentionLayer(nn.Module):

    def __init__(self, condition_dim, embedding_dim, emb_chn, reduction_ratio=1):
        super().__init__()

        self.condition_conv = ResBlock(condition_dim, emb_chn, embedding_dim)
        self.norm_condition1 = BatchNormT(embedding_dim, eps=1e-5)
        self.norm_embedding1 = BatchNormT(embedding_dim, eps=1e-5)

        self.cross_attention1 = CrossAttention(embedding_dim, emb_chn, reduction_ratio)
        self.layerscale1 = LayerScaleModule(embedding_dim, emb_chn)
        self.fnn1 = FNN(embedding_dim, emb_chn)

        self.norm_condition2 = BatchNormT(embedding_dim, eps=1e-5)
        self.norm_embedding2 = BatchNormT(embedding_dim, eps=1e-5)

        self.cross_attention2 = CrossAttention(embedding_dim, emb_chn, reduction_ratio)
        self.layerscale2 = LayerScaleModule(embedding_dim, emb_chn)
        self.fnn2 = FNN(embedding_dim, emb_chn)
        self.apply(self.init_weights)

    def forward(self, condition, embedding, emb):
        # condition: B C1 H W
        # embedding: B C2 H W (C1>C2)
        condition = self.condition_conv(condition, emb)  # B C2 H W
        B, C, H, W = embedding.shape
        patch_num = condition.shape[2] * condition.shape[3]
        condition_embedding_before = condition.flatten(2).transpose(1, 2)
        embedding_embedding_before = embedding.flatten(2).transpose(1, 2)

        condition_embedding = self.norm_condition1(condition_embedding_before)
        embedding_embedding = self.norm_embedding1(embedding_embedding_before)

        condition_embedding = self.cross_attention1(condition_embedding, embedding_embedding, emb)
        attention_output = self.layerscale1(condition_embedding, emb)
        condition_embedding = self.fnn1(condition_embedding_before, attention_output, emb)

        condition_embedding = self.norm_condition2(condition_embedding)
        embedding_embedding = self.norm_embedding2(embedding_embedding)

        embedding_embedding = self.cross_attention2(embedding_embedding, condition_embedding, emb)
        attention_output = self.layerscale2(embedding_embedding, emb)
        embedding_embedding = self.fnn2(embedding_embedding_before, attention_output, emb)

        embedding_embedding = embedding_embedding.reshape(B, C, H, W).contiguous()
        
        return embedding_embedding
    
    @torch.no_grad()
    def init_weights(module: nn.Module, name: str = ''):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm1d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
