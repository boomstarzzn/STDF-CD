import math
from enum import Enum
from typing import Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.cuda.amp import autocast
import einops
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from xlstmnet.xlstmcd import TT_xlstm


#from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param

def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    device = ltr.device  # 获取 ltr 所在的设备
    _log_fg_matrix = _log_fg_matrix.to(device)  # 将 _log_fg_matrix 移动到同一设备

    # 确保 torch.full_like 创建的张量在相同设备上
    inf_tensor = torch.full_like(_log_fg_matrix[:, :, 1:, 1:], -float("inf")).to(device)

    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], inf_tensor)  # (B, NH, S, S)
    #log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)
    # log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:],
    #                             torch.tensor(float('-inf')).to(_log_fg_matrix.dtype))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=False)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=False,
            conv_bias=True,
            kernel_size=4,
    ):
        super().__init__()
        if dim % qkv_block_size != 0:
            qkv_block_size=2
        # assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.conv1d = CausalConv1d(
            dim=inner_dim,
            kernel_size=kernel_size,
            bias=conv_bias,
        )
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape 
        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)


        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)
        #h_state = h_tilde_state_skip * F.silu(y)
        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x
    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()
class ViLLayerce(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=False,
            conv_bias=True,
            kernel_size=4,
    ):
        super().__init__()
        if dim % qkv_block_size != 0:
            qkv_block_size=2
        # assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.projy_up = nn.Linear(
            in_features=dim,
            out_features=2 * dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.conv1d = CausalConv1d(
            dim=inner_dim,
            kernel_size=kernel_size,
            bias=conv_bias,
        )
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor,y) -> torch.Tensor:
        B, S, _ = x.shape 
        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1) 
        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)
        y_in = self.projy_up(y)

        # output / z branch
        #h_state = h_tilde_state_skip * F.silu(z)
        h_state = h_tilde_state_skip * F.silu(y_in)
        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x
    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    def __init__(self, dim, direction, drop_path=0.0, norm_bias=False):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.norm_bias = norm_bias

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(dim=dim, direction=direction)

        self.reset_parameters()

    # def _forward_path(self, x):
    #     x = self.norm(x)
    #     x = self.layer(x)
    #     return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x)
        x = x + self.drop_path(self.layer(x1))
        # print('In xlstm now')
        return x
    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()

class ViLBlockce(nn.Module):
    def __init__(self, dim, direction, drop_path=0.0, norm_bias=False):
        super().__init__()
        self.dim = dim
        self.direction = direction
        #self.drop_path = drop_path
        self.norm_bias = norm_bias
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayerce(dim=dim, direction=direction)
        self.reset_parameters()
    # def _forward_path(self, x,y):
    #
    #     return x,y
    # def _forward_path1(self, x,dif):
    #     x = self.layer(x,dif)
    #     return x
    # def _forward_path2(self, y,dif):
    #     y = self.layer(y,dif)
    #     return y
    def forward(self, x: torch.Tensor,y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        y = self.norm(y)
        dif = torch.abs(y - x)
        #dif = y - x
        x1 = self.layer(x,dif)
        y1 = self.layer(y,dif)
        x = x + self.drop_path(x1) 
        y = y + self.drop_path(y1)
        # print('In xlstm now')
        return x,y

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()
class Vision_xlstmce(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        # print(f"ViLLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlockce(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x,y):
        B, d_model = x.shape[:2]  #B,C
        assert d_model == self.dim #C
        n_tokens = x.shape[2:].numel()  #H*W = L
        img_dims = x.shape[2:]  
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  #B,L,C
        y_flat = y.reshape(B, d_model, n_tokens).transpose(-1, -2)  #B,L,C
        x_vil,y_vil = self.vil(x_flat,y_flat)  #B,L,C


        out = x_vil.transpose(-1, -2).reshape(B, d_model, *img_dims)   #B,C,H,W
        out_y = y_vil.transpose(-1, -2).reshape(B, d_model, *img_dims) #B,C,H,W
        return out,out_y

    def forward_channel_token(self, x,y):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        y_flat = y.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_vil, y_vil= self.vil(x_flat,y_flat)
        #y_vil = self.vil(y_flat)
        out = x_vil.reshape(B, n_tokens, *img_dims)
        out_y = y_vil.reshape(B, n_tokens, *img_dims)
        return out,out_y
    @autocast(enabled=False)
    def forward(self, x,y):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        if y.dtype == torch.float16:
            y = y.type(torch.float32)

        if self.channel_token:
            out,out_y = self.forward_channel_token(x,y)
        else:
            out,out_y = self.forward_patch_token(x,y)

        return out,out_y


class STxlstmce(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # print(f"ViLLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vilce = ViLBlockce(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    def forward(self, x, y):
        B, d_model = x.shape[:2]  # B,C
        assert d_model == self.dim  # C
        n_tokens = x.shape[2:].numel()  # H*W = L
        img_dims = x.shape[2:] 
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # B,L,C
        y_flat = y.reshape(B, d_model, n_tokens).transpose(-1, -2)  # B,L,C
        #SDxlstm
        x_vil, y_vil = self.vilce(x_flat, y_flat)  # B,L,C
        #TT-xlstm
        x_ce = self.norm(x_vil)   # B,L,C
        y_ce = self.norm(y_vil)   # B,L,C
        #resiual
        x_identity = x_ce
        y_identity = y_ce
        B, L, C = x_ce.shape
        img_fuse1 = x_ce.permute(0, 2, 1).unsqueeze(-1)  # (B,C,H*W(L),1)
        img_fuse2 = y_ce.permute(0, 2, 1).unsqueeze(-1)  # (B,C,H*W(L),1)
        img_fuse = torch.cat([img_fuse1, img_fuse2], dim=-1).reshape(B, C, -1)  # (B,C,L*2)
        img_fuse = self.vil(img_fuse.permute(0, 2, 1)) # (B,C,L*2)
        img_fuse = img_fuse.reshape(B, C, L, -1)
        x_ce = img_fuse[..., 0].permute(0, 2, 1)  # [...,:C] # (B,L,C)
        y_ce = img_fuse[..., 1].permute(0, 2, 1)  # [...,:C] # (B,L,C)

        x_ce = self.norm(x_ce) + x_identity * self.alpha
        y_ce = self.norm(y_ce) + y_identity * self.alpha

        out = x_ce.transpose(-1, -2).reshape(B, d_model, *img_dims)  # B,C,H,W
        out_y = y_ce.transpose(-1, -2).reshape(B, d_model, *img_dims)  # B,C,H,W
        return out, out_y

class STxlstmfs(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.vilce = ViLBlockce(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.mlp1 = ConvolutionalGLU(in_features=dim * 2, hidden_features=4* dim,out_features=dim, act_layer=nn.GELU, drop=0)
        self.drop_path = DropPath(drop_prob=0)

    def forward(self, x, y, H, W):
        B, C = x.shape[:2]  # B, C
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        y_flat = y.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        # ViLBlockce
        x_vil, y_vil = self.vilce(x_flat, y_flat)  # (B, H*W, C)
        # Normalization
        x_ce = self.norm(x_vil)  # (B, H*W, C)
        y_ce = self.norm(y_vil)  # (B, H*W, C)
        # Concatenate features along the channel dimension
        x_fused = torch.cat((x_ce, y_ce), dim=-1)  # (B, H*W, 2*C)
        # Apply MLP to perform the mixing operation
        x_out = self.mlp1(x_fused, H, W)  # (B, H*W, 2*C) -> (B, H*W, C)
        # Apply DropPath and final normalization
        x_out = self.norm3(x_out).view(B, H, W, C).permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        return x_out

class MultiLayerDendriteFusion(nn.Module):
    def __init__(self, input_size, out_size, activation=F.relu):
        super(MultiLayerDendriteFusion, self).__init__()
        self.num_layers = 4  
        #self.qs = nn.Parameter(torch.rand(1))
        self.params = nn.ParameterDict({
            'DNM_W1': nn.Parameter(torch.rand([out_size, input_size])),
            'DNM_W2': nn.Parameter(torch.rand([out_size, input_size])),
            'DNM_W3': nn.Parameter(torch.rand([out_size, input_size])),
            'DNM_W4': nn.Parameter(torch.rand([out_size, input_size])),
            'q1': nn.Parameter(torch.rand([out_size, input_size])),
            'q2': nn.Parameter(torch.rand([out_size, input_size])),
            'q3': nn.Parameter(torch.rand([out_size, input_size])),
            'q4': nn.Parameter(torch.rand([out_size, input_size]))
        })
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.activation = activation
        self.conv_de = nn.Conv2d(input_size * 4, out_size, 1)

    def forward(self, x1, x2, x3, x4):
        x1 = self.process_layer(x1, 'DNM_W1', 'q1')
        x2 = self.process_layer(x2, 'DNM_W2', 'q2')
        x3 = self.process_layer(x3, 'DNM_W3', 'q3')
        x4 = self.process_layer(x4, 'DNM_W4', 'q4')
        #x = x1 + x2 + x3 + x4
        x = self.conv_de(torch.cat((x1, x2, x3, x4), dim=1))

        if self.activation:
            #x = self.activation(x-self.qs)
            x = self.activation(x)
        return x

    def process_layer(self, x, weight_key, q_key):

        x = x.permute(0, 2, 3, 1)  # [batch_size, C, H, W] -> [batch_size, H, W, C]
        x = self.norm1(x)  
        x = F.relu(torch.mul(x, self.params[weight_key]) - self.params[q_key])  
        x = self.norm2(x) 
        x = x.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch_size, C, H, W]
        return x
class Mconv_dnm(nn.Module):
    def __init__(self, in_channal, out_channal):
        super(Mconv_dnm, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.Dnm_conv2d = MultiLayerDendriteFusion(in_channal, out_channal, activation=None)
        # self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.Dnm_conv2d(x)
        return x




class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


