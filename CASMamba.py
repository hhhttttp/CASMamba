import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from .mamba.multi_mamba import MultiScan

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    if torch.__version__ > '2.0.0':
        from selective_scan_vmamba_pt202 import selective_scan_cuda_core
    else:
        from selective_scan_vmamba import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda


# ===========================================================================
#  Helper Functions (FLOPS, Selective Scan)
# ===========================================================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    assert not with_complex
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if C.stride(-1) != 1: C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1: dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


# ===========================================================================
#  Attention & Basic Modules
# ===========================================================================

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_sigmoid()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


# ===========================================================================
#  Paper Specific Modules: CPM & PFAG
# ===========================================================================

# Renamed ESTD -> CPM (Coordinate Perception Module)
class CPM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CPM, self).__init__()
        # Using CoordAtt as the core of CPM
        self.coord_attention = CoordAtt(in_channels, in_channels, reduction=reduction_ratio)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        residual = x
        attended_x = self.coord_attention(x)
        projected_x = self.projection(attended_x)
        out = residual + projected_x
        return out


# Renamed CARG -> PFAG (Parameter-Free Attention Gate)
class PFAG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(PFAG, self).__init__()
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      groups=in_channels, bias=False)
        )
        # Using SimAM as the core of PFAG
        self.attention = SimAM()
        self.activation = nn.SiLU(inplace=True)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_res = self.residual_conv(x)
        x_main = self.main_path(x)
        x_att = self.attention(x_main)
        # Refine features
        x_att_activated = self.activation(x_att + x_main)
        gated_features = x_att_activated * x_res
        out = self.final_conv(gated_features)
        return out


# ===========================================================================
#  Scanning Mechanism: CAS (Content-Aware Scan)
# ===========================================================================

class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)

        attn = c_attn * s_attn
        out = ori_x * attn
        return out


# Renamed MultiScanVSSM -> ContentAwareScan
class ContentAwareScan(MultiScan):
    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None, win_size=8):
        super().__init__(dim, choices=choices, token_size=None, win_size=win_size)
        self.attn = BiAttn(dim)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [x.transpose(-2, -1) for x in xs]
        x = super().forward(xs)
        return x

    def multi_scan(self, x):
        # x: [B, C, H, W] -> [B, K, C, L]
        B, C, H, W = x.shape
        self.token_size = (H, W)
        xs = super().multi_scan(x)
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # Pad tokens to same length for parallel computing
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('ContentAwareScan', f'ContentAwareScan[{scans}]')


def multi_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        multi_scan=None,
        win_size=8,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    # Perform Content-Aware Scan
    xs = multi_scan.multi_scan(x)

    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    y = multi_scan.merge(ys)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


# ===========================================================================
#  SS2D Module (Core Mamba Integration)
# ===========================================================================

class SS2D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            directions=None,
            win_size=8,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)
        self.K = len(ContentAwareScan.ALL_CHOICES) if directions is None else len(directions)
        self.K2 = self.K

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)

        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.win_size = win_size

        # Initialize Content Aware Scan Mechanism
        self.multi_scan = ContentAwareScan(d_expand, choices=directions, win_size=self.win_size)

        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = multi_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan, win_size=self.win_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ===========================================================================
#  CAS Block (Content-Aware Scan Block)
# ===========================================================================

# Renamed VSSBlock -> CASBlock
class CASBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            win_size=8,
            use_checkpoint: bool = False,
            directions=None,
            # Renamed args to match paper modules
            use_cpm: bool = False,  # was use_estd
            use_pfag: bool = False,  # was use_carg
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_cpm = use_cpm
        self.use_pfag = use_pfag

        # --- Stage 1: Spatial Perception & Context-Aware Scan ---
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank, act_layer=ssm_act_layer, d_conv=ssm_conv,
            conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
            simple_init=ssm_simple_init, directions=directions, win_size=win_size,
        )
        self.drop_path = DropPath(drop_path)

        if self.use_cpm:
            self.cpm = CPM(in_channels=hidden_dim)

        # --- Stage 2: Feature Rectification (PFAG) ---
        if self.use_pfag:
            self.norm2 = norm_layer(hidden_dim)
            self.pfag = PFAG(in_channels=hidden_dim)

    def _forward(self, input: torch.Tensor):
        # --- Stage 1: CPM + SS2D (CAS) ---
        residual1 = input
        norm1_x = self.norm(input)

        if self.use_cpm:
            # Inject spatial priors via Coordinate Perception Module
            norm1_x_cf = norm1_x.permute(0, 3, 1, 2).contiguous()
            cpm_out_cf = self.cpm(norm1_x_cf)
            processed_input = cpm_out_cf.permute(0, 2, 3, 1).contiguous()
        else:
            processed_input = norm1_x

        ss2d_out = self.op(processed_input)
        x = residual1 + self.drop_path(ss2d_out)

        # --- Stage 2: PFAG Refinement ---
        if self.use_pfag:
            residual2 = x
            norm2_x = self.norm2(x)

            norm2_x_cf = norm2_x.permute(0, 3, 1, 2).contiguous()
            refined_out_cf = self.pfag(norm2_x_cf)
            refined_out = refined_out_cf.permute(0, 2, 3, 1).contiguous()

            out = residual2 + self.drop_path(refined_out)
            return out
        else:
            return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# ===========================================================================
#  CASMamba Network Architecture
# ===========================================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


# Renamed VSSM -> CASMamba
class CASMamba(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 2, 2],
            dims=[96, 192, 384, 768],
            depths_decoder=[2, 2, 2, 1],
            dims_decoder=[768, 384, 192, 96],
            win_size=[8, 4, 2, 1],
            decoder_win_size=[1, 2, 4, 8],
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            # Configs for Paper Modules
            use_cpm: bool = False,  # was use_estd
            use_pfag: bool = False,  # was use_carg
            use_checkpoint=False,
            directions=None,
            decoder_directions=None,
            nr_types=None, freeze=False,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.dim_decoder = dims_decoder
        self.freeze = freeze
        self.nr_types = nr_types
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        decoder_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]

        _NORMLAYERS = dict(ln=nn.LayerNorm, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]
        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(dims[0]) if patch_norm else nn.Identity()),
        )

        # Encoder Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging2D(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                win_size=win_size[i_layer],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_cpm=use_cpm,
                use_pfag=use_pfag,
                directions=None if directions is None else directions[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            ))

        # Decoder Layers (Branch 1: Nuclei Prediction)
        self.layers_up_np = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up_np.append(self._make_layer_up(
                dim=self.dim_decoder[i_layer],
                drop_path=decoder_dpr[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=PatchExpand2D(dim=self.dim_decoder[i_layer], dim_scale=2) if (i_layer != 0) else None,
                win_size=decoder_win_size[i_layer],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_cpm=use_cpm,
                use_pfag=use_pfag,
                directions=None if decoder_directions is None else decoder_directions[sum(depths_decoder[:i_layer]):sum(
                    depths_decoder[:i_layer + 1])]
            ))
        self.final_up_np = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)

        # Decoder Layers (Branch 2: Type/Distance Prediction if applicable)
        self.layers_up_tp = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up_tp.append(self._make_layer_up(
                dim=self.dim_decoder[i_layer],
                drop_path=decoder_dpr[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=PatchExpand2D(dim=self.dim_decoder[i_layer], dim_scale=2) if (i_layer != 0) else None,
                win_size=decoder_win_size[i_layer],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_cpm=use_cpm,
                use_pfag=use_pfag,
                directions=None if decoder_directions is None else decoder_directions[sum(depths_decoder[:i_layer]):sum(
                    depths_decoder[:i_layer + 1])]
            ))
        self.final_up_tp = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)

        if self.nr_types is None:
            self.final_conv_np = nn.Conv2d(dims_decoder[-1] // 4, 4, 1)
        else:
            self.final_conv_tp = nn.Conv2d(dims_decoder[-1] // 4, nr_types, 1)
            self.final_conv_np = nn.Conv2d(dims_decoder[-1] // 4, 4, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            win_size=8,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            use_cpm=False,
            use_pfag=False,
            directions=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(CASBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                win_size=win_size,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_checkpoint=use_checkpoint,
                directions=directions[d] if directions is not None else None,
                use_cpm=use_cpm,
                use_pfag=use_pfag
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    @staticmethod
    def _make_layer_up(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            upsample=None,
            win_size=8,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            use_cpm=False,
            use_pfag=False,
            directions=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(CASBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                win_size=win_size,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                use_checkpoint=use_checkpoint,
                directions=directions[d] if directions is not None else None,
                use_cpm=use_cpm,
                use_pfag=use_pfag
            ))
        if upsample is None:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks, )
            ))
        if upsample is not None:
            return nn.Sequential(OrderedDict(
                upsample=upsample,
                blocks=nn.Sequential(*blocks, ),
            ))

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward(self, x):
        if self.training:
            with torch.set_grad_enabled(not self.freeze):
                b_out, skip_list = self.forward_features(x)
        else:
            b_out, skip_list = self.forward_features(x)

        out_dict = OrderedDict()
        if self.nr_types is None:
            for inx, layer_up in enumerate(self.layers_up_np):
                if inx == 0:
                    x = layer_up(b_out)
                else:
                    x = layer_up(x + skip_list[-inx])

            x = self.final_up_np(x)
            x = x.permute(0, 3, 1, 2)
            x = self.final_conv_np(x)
            out_dict['np'] = x
        else:
            for branch_name in ['tp', 'np']:
                layers_up = getattr(self, f'layers_up_{branch_name}')
                for inx, layer_up in enumerate(layers_up):
                    if inx == 0:
                        x = layer_up(b_out)
                    else:
                        x = layer_up(x + skip_list[-inx])
                final_up = getattr(self, f'final_up_{branch_name}')
                x = final_up(x)
                x = x.permute(0, 3, 1, 2)
                final_conv = getattr(self, f'final_conv_{branch_name}')
                x = final_conv(x)
                out_dict[branch_name] = x

        return out_dict

    def flops(self, shape=(3, 224, 224)):
        # Function to calculate flops, assuming external dependency is handled
        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }
        model = copy.deepcopy(self)
        model.cuda().eval()
        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        # Note: Need fvcore installed for this to work
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
            del model, input
            return sum(Gflops.values()) * 1e9
        except ImportError:
            print("fvcore not found, cannot calculate FLOPs")
            return 0


# ===========================================================================
#  Model Registration
# ===========================================================================

@register_model
def cas_mamba(*args,
              in_chans=3,
              dims=[96, 192, 384, 768],
              depths=[2, 2, 2, 2],
              depths_decoder=[2, 2, 2, 1],
              dims_decoder=[768, 384, 192, 96],
              d_state=16,
              drop_path_rate=0.2,
              freeze=False,
              nr_types=None,
              use_cpm: bool = False,  # Renamed from use_estd
              use_pfag: bool = False,  # Renamed from use_carg
              **kwargs):
    # --- Encoder Directions ---
    directions = [
        # Layer 0 (depth=2)
        ['v', 'h', 'h_flip', 'cas16_flip'],
        ['v_flip', 'cas16_flip', 'h_flip', 'h'],
        # Layer 1 (depth=2)
        ['h', 'cas16_flip', 'v_flip', 'h_flip'],
        ['v_flip', 'h', 'h_flip', 'cas16_flip'],
        # Layer 2 (depth=2)
        ['h', 'h_flip', 'v_flip', 'cas16_flip'],
        ['h', 'cas16_flip', 'v_flip', 'h_flip'],
        # Layer 3 (depth=2)
        ['cas16_flip', 'h', 'v_flip', 'h_flip'],
        ['cas4_flip', 'h_flip', 'cas16', 'cas4'],
    ]

    # --- Decoder Directions ---
    decoder_directions = [
        # Decoder Layer 0 (from layers_up_np.0, depth=2)
        ['h', 'h_flip', 'cas4', 'v_flip'],
        ['v_flip', 'v', 'cas16_flip', 'cas4'],
        # Decoder Layer 1 (from layers_up_np.1, depth=2)
        ['h', 'h_flip', 'v', 'cas16_flip'],
        ['v_flip', 'h', 'cas4', 'cas16'],
        # Decoder Layer 2 (from layers_up_np.2, depth=2)
        ['v_flip', 'v', 'cas16_flip', 'h_flip'],
        ['v', 'h_flip', 'v_flip', 'cas4'],
        # Decoder Layer 3 (from layers_up_np.3, depth=1)
        ['v_flip', 'h_flip', 'h', 'v'],
    ]

    return CASMamba(
        in_chans=in_chans,
        dims=dims,
        depths=depths,
        dims_decoder=dims_decoder,
        depths_decoder=depths_decoder,
        d_state=d_state,
        drop_path_rate=drop_path_rate,
        directions=directions,
        decoder_directions=decoder_directions,
        freeze=freeze,
        nr_types=nr_types,
        use_cpm=use_cpm,
        use_pfag=use_pfag
    )


if __name__ == '__main__':
    from thop import profile

    # Example instantiation
    model = cas_mamba(num_classes=3)
    model.cuda()

    # Print parameters to verify names
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    input = torch.randn(1, 3, 256, 256).cuda()
    out = model(input)

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops / 10 ** 9} G")
    print(f"Parameters: {params / 10 ** 6} M")