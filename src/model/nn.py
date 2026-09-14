import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import Config


# --------------------------- helpers ---------------------------

def _safe_groupnorm(num_groups: int, num_channels: int) -> nn.GroupNorm:
    num_groups = max(1, min(num_groups, num_channels))
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


# --------------------------- building blocks ---------------------------

class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, heads: int, norm_groups: int):
        super().__init__()
        self.heads = max(1, heads if channels % heads == 0 else 1)
        self.norm = _safe_groupnorm(norm_groups, channels)
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        h = self.heads
        x_n = self.norm(x)
        q, k, v = self.to_qkv(x_n).chunk(3, dim=1)
        c_h = c // h
        q = q.view(b, h, c_h, l)
        k = k.view(b, h, c_h, l)
        v = v.view(b, h, c_h, l)
        scale = c_h ** -0.5
        attn = torch.einsum("bhcl,bhcm->bhlm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhlm,bhcm->bhcl", attn, v).contiguous().view(b, c, l)
        return x + self.proj(out)


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        norm_groups: int,
        with_attn: bool,
        attn_heads: int,
        global_cond_dim: int,
    ):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch

        self.norm1 = _safe_groupnorm(norm_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.film = nn.Linear(global_cond_dim, 2 * out_ch)

        self.norm2 = _safe_groupnorm(norm_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.attn = SelfAttention1D(out_ch, attn_heads, norm_groups) if with_attn else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        gamma, beta = self.film(cond_emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        h = gamma * h + beta

        h = self.conv2(F.silu(self.norm2(h)))
        h = h + self.skip(x)
        h = self.attn(h)
        return h


class Downsample1D(nn.Module):
    def __init__(self, ch: int, k: int, s: int, p: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, kernel_size=k, stride=s, padding=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int, k: int, s: int, p: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, kernel_size=k, stride=s, padding=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# --------------------------- UNet1D model ---------------------------

class UNet1D(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        m = config.unet_model

        self.seq_len = m.sequence_size
        self.harm_pow = m.harmonic_positional_encoding_power
        self.t_layers = m.t_encoder_num_layers
        self.p_layers = m.xpos_encoder_num_layers
        self.use_input_posenc = m.use_input_positional_encoding

        base_ch        = m.base_channels
        channel_mults  = m.channel_mults
        num_res_blocks = m.num_res_blocks
        norm_groups    = m.norm_groups
        attn_resols    = set(m.attn_resolutions)
        attn_heads     = m.attn_heads
        self.use_final_tanh = m.final_tanh

        k_down, s_down, p_down = m.down_kernel, m.down_stride, m.down_padding
        k_up,   s_up,   p_up   = m.up_kernel,   m.up_stride,   m.up_padding

        self.extra_cond_dim = m.extra_cond_dim
        self.global_cond_dim = m.global_cond_dim

        pos_dim = 1 + 2 * self.harm_pow
        time_dim = base_ch * 4
        t_layers = [nn.Linear(pos_dim, time_dim)]
        for _ in range(max(self.t_layers - 1, 0)):
            t_layers += [nn.ReLU(), nn.Linear(time_dim, time_dim)]
        self.t_encoder = nn.Sequential(*t_layers)

        merge_in = time_dim + self.extra_cond_dim
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(merge_in, self.global_cond_dim),
        )

        self.null_cond = nn.Parameter(torch.zeros(self.global_cond_dim))

        if self.use_input_posenc:
            self.register_buffer("xpos_coords", torch.linspace(0.0, 1.0, self.seq_len), persistent=False)
            p_layers = [nn.Linear(pos_dim, base_ch)]
            for _ in range(max(self.p_layers - 1, 0)):
                p_layers += [nn.ReLU(), nn.Linear(base_ch, base_ch)]
            self.xpos_encoder = nn.Sequential(*p_layers)
        else:
            self.xpos_encoder = None
            self.register_buffer("xpos_coords", torch.empty(0), persistent=False)

        in_ch = (2 if m.include_other_domain else 1) * self.config.data.data_channels
        self.in_proj  = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.out_norm = _safe_groupnorm(norm_groups, base_ch)
        self.out_proj = nn.Conv1d(base_ch, self.config.data.data_channels, kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()
        self.up_stages   = nn.ModuleList()

        ch = base_ch
        cur_len = self.seq_len
        self.skip_channels = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock1D(
                        in_ch=ch,
                        out_ch=out_ch,
                        time_dim=time_dim,
                        norm_groups=norm_groups,
                        with_attn=(cur_len in attn_resols),
                        attn_heads=attn_heads,
                        global_cond_dim=self.global_cond_dim,
                    )
                )
                ch = out_ch
                self.skip_channels.append(out_ch)
            downsample = None
            if i != len(channel_mults) - 1:
                downsample = Downsample1D(ch, k_down, s_down, p_down)
                cur_len = (cur_len + 2 * p_down - k_down) // s_down + 1
            self.down_stages.append(nn.ModuleDict({"blocks": blocks, "down": downsample}))

        self.mid_block1 = ResBlock1D(
            ch, ch, time_dim, norm_groups,
            with_attn=True,
            attn_heads=attn_heads,
            global_cond_dim=self.global_cond_dim,
        )
        self.mid_block2 = ResBlock1D(
            ch, ch, time_dim, norm_groups,
            with_attn=False,
            attn_heads=attn_heads,
            global_cond_dim=self.global_cond_dim,
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(
                    ResBlock1D(
                        in_ch=ch + skip_ch,
                        out_ch=out_ch,
                        time_dim=time_dim,
                        norm_groups=norm_groups,
                        with_attn=(cur_len in attn_resols),
                        attn_heads=attn_heads,
                        global_cond_dim=self.global_cond_dim,
                    )
                )
                ch = out_ch
            upsample = None
            if i != 0:
                upsample = Upsample1D(ch, k_up, s_up, p_up)
                cur_len = (cur_len - 1) * s_up - 2 * p_up + k_up
            self.up_stages.append(nn.ModuleDict({"blocks": blocks, "up": upsample}))

    def harmonic_positional_encoding(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t[:, None]
        freqs = []
        for i in range(self.harm_pow):
            w = 2.0 ** i
            freqs.extend([torch.sin(t * w), torch.cos(t * w)])
        return torch.cat([t] + freqs, dim=-1)

    def harmonic_positional_encoding_1d_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 1:
            coords = coords[:, None]
        freqs = []
        for i in range(self.harm_pow):
            w = 2.0 ** i
            freqs.extend([torch.sin(coords * w), torch.cos(coords * w)])
        return torch.cat([coords] + freqs, dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        if isinstance(t, float) or len(t.shape) == 0:
            t = torch.zeros(x.shape[0], 1, dtype=torch.float32, device=x.device) + t
        t = t.reshape(-1,1)

        b, c, d = x.shape
        assert d == self.seq_len, f"Expected input dim {self.seq_len}, got {d}"

        t_pe = self.harmonic_positional_encoding(t.to(dtype=x.dtype, device=x.device))
        t_emb = self.t_encoder(t_pe)

        if self.extra_cond_dim > 0:
            if cond is None:
                cond_vec = self.null_cond.unsqueeze(0).expand(b, -1)
            else:
                assert cond.shape[0] == b
                assert cond.shape[1] == self.extra_cond_dim
                cond_vec = cond
            merged = torch.cat([t_emb, cond_vec.to(t_emb.dtype)], dim=1)
        else:
            cond_vec = self.null_cond.unsqueeze(0).expand(b, -1)
            merged = torch.cat([t_emb, cond_vec.new_zeros(b, 0)], dim=1)

        cond_emb = self.cond_mlp(merged)

        h = self.in_proj(x)
        if self.use_input_posenc:
            coords = self.xpos_coords.to(device=x.device, dtype=x.dtype)
            pos_pe = self.harmonic_positional_encoding_1d_coords(coords)
            pos_emb = self.xpos_encoder(pos_pe)
            pos_emb = pos_emb.transpose(0, 1).unsqueeze(0).contiguous()
            h = h + pos_emb

        skips = []
        for stage in self.down_stages:
            for block in stage["blocks"]:
                h = block(h, cond_emb)
                skips.append(h)
            if stage["down"] is not None:
                h = stage["down"](h)

        h = self.mid_block1(h, cond_emb)
        h = self.mid_block2(h, cond_emb)

        for stage in self.up_stages:
            for block in stage["blocks"]:
                skip = skips.pop()
                if skip.shape[-1] != h.shape[-1]:
                    diff = skip.shape[-1] - h.shape[-1]
                    if diff > 0:
                        h = F.pad(h, (0, diff))
                    else:
                        skip = F.pad(skip, (0, -diff))
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond_emb)
            if stage["up"] is not None:
                h = stage["up"](h)

        h = self.out_proj(F.silu(self.out_norm(h)))
        if self.use_final_tanh:
            h = torch.tanh(h)
        return h


# --------------------------- Encoder1D ---------------------------

class _ResBlock1DNoCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_groups: int):
        super().__init__()
        self.norm1 = _safe_groupnorm(norm_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = _safe_groupnorm(norm_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder1D(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        m = config.unet_model

        self.seq_len = m.sequence_size

        base_ch        = m.base_channels
        channel_mults  = m.channel_mults
        num_res_blocks = m.num_res_blocks
        norm_groups    = m.norm_groups

        k_down, s_down, p_down = m.down_kernel, m.down_stride, m.down_padding

        in_ch = (2 if m.include_other_domain else 1) * config.data.data_channels
        self.in_proj = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()

        ch = base_ch
        cur_len = self.seq_len

        for i, mult in enumerate(channel_mults):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks.append(_ResBlock1DNoCond(ch, out_ch, norm_groups))
                ch = out_ch

            downsample = None
            if i != len(channel_mults) - 1:
                downsample = Downsample1D(ch, k_down, s_down, p_down)
                cur_len = (cur_len + 2 * p_down - k_down) // s_down + 1

            self.down_stages.append(nn.ModuleDict({"blocks": blocks, "down": downsample}))

        # ---- positional encoding (same as UNet input posenc) ----
        self.harm_pow = m.harmonic_positional_encoding_power
        self.p_layers = m.xpos_encoder_num_layers
        self.use_input_posenc = m.use_input_positional_encoding

        pos_dim = 1 + 2 * self.harm_pow
        if self.use_input_posenc:
            self.register_buffer("xpos_coords", torch.linspace(0.0, 1.0, self.seq_len), persistent=False)
            p_layers = [nn.Linear(pos_dim, base_ch)]
            for _ in range(max(self.p_layers - 1, 0)):
                p_layers += [nn.ReLU(), nn.Linear(base_ch, base_ch)]
            self.xpos_encoder = nn.Sequential(*p_layers)
        else:
            self.xpos_encoder = None
            self.register_buffer("xpos_coords", torch.empty(0), persistent=False)

        self.out_norm = _safe_groupnorm(norm_groups, ch)
        self.out_proj = nn.Linear(ch * cur_len, m.extra_cond_dim)
        self.contrastive_divergence_temp = nn.Parameter(torch.tensor(0.5))

    def harmonic_positional_encoding_1d_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 1:
            coords = coords[:, None]
        freqs = []
        for i in range(self.harm_pow):
            w = 2.0 ** i
            freqs.extend([torch.sin(coords * w), torch.cos(coords * w)])
        return torch.cat([coords] + freqs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)

        if self.use_input_posenc:
            coords = self.xpos_coords.to(device=x.device, dtype=x.dtype)
            pos_pe = self.harmonic_positional_encoding_1d_coords(coords)
            pos_emb = self.xpos_encoder(pos_pe)                      # (L, base_ch)
            pos_emb = pos_emb.transpose(0, 1).unsqueeze(0).contiguous()  # (1, base_ch, L)
            h = h + pos_emb

        for stage in self.down_stages:
            for block in stage["blocks"]:
                h = block(h)
            if stage["down"] is not None:
                h = stage["down"](h)

        h = F.silu(self.out_norm(h))
        h = h.flatten(1)
        return self.out_proj(h)
    

