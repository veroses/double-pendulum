from dataclasses import dataclass, field

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
@dataclass
class MLPEncoderConfig:
    layer_sizes: list[int] = field(default_factory=list)  # [] = identity
    activation: str = "relu"


@dataclass
class VisionEncoderConfig:
    """Config for the multi-view ResNet + Transformer vision encoder.

    The backbone is a small residual CNN whose weights are shared across the two
    camera views; its feature map is tokenized and fused with a Transformer
    encoder. Only architectural hyperparameters live here — input/output sizes
    that are derivable from the data (e.g. the torque dimension, the GPT token
    dimension) are passed in at construction time, not hardcoded.
    """
    image_size: int = 64
    in_channels: int = 3
    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (64, 128, 128)
    fusion_layers: int = 2
    fusion_heads: int = 4
    fusion_dim: int = 128
    out_dim: int = 192
    dropout: float = 0.0
    norm: str = "group"
    norm_groups: int = 8


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _make_norm(norm: str, num_channels: int, num_groups: int) -> nn.Module:
    """GroupNorm by default (BatchNorm is avoided for small-batch robustness)."""
    if norm == "group":
        assert num_channels % num_groups == 0, (
            f"norm_groups={num_groups} does not divide channels={num_channels}"
        )
        return nn.GroupNorm(num_groups, num_channels)
    raise NotImplementedError(f"unsupported norm: {norm!r} (only 'group')")


class ResBlock(nn.Module):
    """Downsampling residual block: two 3x3 convs (first strided) + 1x1 skip.

    GroupNorm + SiLU throughout. The skip is a stride-2 1x1 conv projection so
    the residual matches the strided/channel-changed main path.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, cfg: VisionEncoderConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _make_norm(cfg.norm, out_ch, cfg.norm_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = _make_norm(cfg.norm, out_ch, cfg.norm_groups)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.skip(x))


class VisionBackbone(nn.Module):
    """Per-view residual CNN. [B, 3, 64, 64] -> [B, num_tokens, fusion_dim] tokens.

    Stem (3x3, stride 1) keeps resolution, then three stride-2 ResBlocks halve
    it each time: 64 -> 32 -> 16 -> 8. The final [B, C, 8, 8] map is tokenized
    into [B, 64, C] with flatten(2).transpose(1,2) so that each token is one
    spatial location's channel vector (never interleaving channels and space).
    """

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.stage_channels[-1] == cfg.fusion_dim, (
            f"last stage channels {cfg.stage_channels[-1]} must equal "
            f"fusion_dim {cfg.fusion_dim} (token embedding dim)"
        )

        self.stem = nn.Conv2d(
            cfg.in_channels, cfg.stem_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_norm = _make_norm(cfg.norm, cfg.stem_channels, cfg.norm_groups)
        self.act = nn.SiLU()

        blocks = []
        in_ch = cfg.stem_channels
        for out_ch in cfg.stage_channels:
            blocks.append(ResBlock(in_ch, out_ch, stride=2, cfg=cfg))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Spatial size after stem (stride 1) + one /2 per stage.
        self.grid_size = cfg.image_size // (2 ** len(cfg.stage_channels))
        self.num_tokens = self.grid_size * self.grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        assert x.shape[1:] == (self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size), x.shape

        x = self.act(self.stem_norm(self.stem(x)))
        x = self.blocks(x)  # [B, fusion_dim, grid, grid]
        assert x.shape == (B, self.cfg.fusion_dim, self.grid_size, self.grid_size), x.shape

        # [B, C, H, W] -> [B, H*W, C]: token i is spatial location i's channels.
        tokens = x.flatten(2).transpose(1, 2)
        assert tokens.shape == (B, self.num_tokens, self.cfg.fusion_dim), tokens.shape
        return tokens


class MultiViewFusion(nn.Module):
    """Fuse two camera views into a single per-timestep feature vector.

    The shared backbone tokenizes each view; tokens get a shared 2D positional
    embedding (over the 8x8 grid) plus a per-view embedding, are concatenated,
    and a learned CLS token is prepended. A pre-LN Transformer encoder mixes
    them and the CLS output is projected to out_dim.
    """

    NUM_VIEWS = 2

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = VisionBackbone(cfg)
        P, D = self.backbone.num_tokens, cfg.fusion_dim

        # Positional embedding shared across views (one per 8x8 grid position).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, P, D))
        # One embedding per view (cam0 vs cam1).
        self.view_embed = nn.Parameter(torch.zeros(1, self.NUM_VIEWS, 1, D))
        # Learned CLS token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg.fusion_heads,
            dim_feedforward=4 * D,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.fusion_layers, enable_nested_tensor=False
        )
        self.proj = nn.Linear(D, cfg.out_dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # views: [B, 2, C, H, W]
        B, V = views.shape[:2]
        assert V == self.NUM_VIEWS, views.shape
        C, H, W = views.shape[2:]
        P, D = self.backbone.num_tokens, self.cfg.fusion_dim

        # Shared backbone over both views: fold view into batch.
        x = views.reshape(B * V, C, H, W)
        tokens = self.backbone(x)                       # [B*V, P, D]
        tokens = tokens.reshape(B, V, P, D)             # [B, V, P, D]
        assert tokens.shape == (B, V, P, D), tokens.shape

        tokens = tokens + self.pos_embed + self.view_embed
        # Concat views along the token axis: view0's P tokens then view1's P.
        tokens = tokens.reshape(B, V * P, D)
        assert tokens.shape == (B, V * P, D), tokens.shape

        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)        # [B, 1 + V*P, D]

        tokens = self.encoder(tokens)
        cls_out = tokens[:, 0]                           # [B, D]
        feat = self.proj(cls_out)                        # [B, out_dim]
        assert feat.shape == (B, self.cfg.out_dim), feat.shape
        return feat


class MLPEncoder(nn.Module):
    """State encoder for the (non-vision) dynamics model.

    Counterpart to VisionEncoder for the case where the model is fed the raw
    MuJoCo state rather than camera frames. Essentially the identity function:

      * Constructed as ``MLPEncoder(input_size, config)`` where ``input_size`` is
        the per-timestep state width (a data property, not hardcoded).
      * ``config.layer_sizes == []`` -> the encoder is the **identity**: it
        returns its input unchanged and ``output_dim == input_size``.
      * ``config.layer_sizes == [..., d]`` -> an MLP with ``config.activation``
        between layers (none on the output), projecting to ``d``; ``output_dim
        == d``.
      * Exposes an ``output_dim`` attribute (the width handed to the dynamics
        core) and preserves all leading batch/time dims: ``[..., input_size] ->
        [..., output_dim]``.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(self, input_size: int, config: MLPEncoderConfig):
        super().__init__()
        if config.activation not in self._ACTIVATIONS:
            raise NotImplementedError(f"unsupported activation: {config.activation!r}")
        act = self._ACTIVATIONS[config.activation]

        if not config.layer_sizes:
            # Identity: pass the raw state straight through to the dynamics core.
            self.net = nn.Identity()
            self.output_dim = input_size
            return

        layers = []
        prev = input_size
        for i, size in enumerate(config.layer_sizes):
            layers.append(nn.Linear(prev, size))
            if i < len(config.layer_sizes) - 1:  # no activation on the output
                layers.append(act())
            prev = size
        self.net = nn.Sequential(*layers)
        self.output_dim = config.layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear / nn.Identity act on the last dim, so leading dims are kept.
        return self.net(x)


class VisionEncoder(nn.Module):
    """Sequence-level encoder feeding the (separate) causal dynamics transformer.

    Input frames are [B, T, 2, 3, 64, 64]: two camera views per timestep over a
    history window T. T is folded into the batch, each timestep's two views are
    fused to a vision feature [B, T, out_dim], then (optionally) the per-timestep
    torque is concatenated and projected to produce the token sequence the
    dynamics GPT consumes. The GPT and Δs head live elsewhere and are not built
    here (the dynamics module is left untouched).
    """

    def __init__(self, cfg: VisionEncoderConfig, torque_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.torque_dim = torque_dim
        self.fusion = MultiViewFusion(cfg)
        # Maps concat[vision feat, torque] -> token. The torque dimension is a
        # data property (passed in, not hardcoded); the token dimension equals
        # out_dim (the GPT's n_embd). Built only when torque is actually used.
        self.token_proj = (
            nn.Linear(cfg.out_dim + torque_dim, cfg.out_dim) if torque_dim is not None else None
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"VisionEncoder initialized | {n_params:,} parameters")

    def forward(self, frames: torch.Tensor, torque: torch.Tensor | None = None) -> torch.Tensor:
        # frames: [B, T, 2, C, H, W]
        B, T = frames.shape[:2]
        V, C, H, W = frames.shape[2:]
        assert (V, C, H, W) == (
            MultiViewFusion.NUM_VIEWS, self.cfg.in_channels, self.cfg.image_size, self.cfg.image_size
        ), frames.shape

        # Fold T into the batch so the fusion module sees per-timestep [*, 2, C, H, W].
        x = frames.reshape(B * T, V, C, H, W)
        feat = self.fusion(x)                            # [B*T, out_dim]
        feat = feat.reshape(B, T, self.cfg.out_dim)      # [B, T, out_dim]
        assert feat.shape == (B, T, self.cfg.out_dim), feat.shape

        if torque is None:
            return feat                                  # vision features only

        assert self.token_proj is not None, "construct VisionEncoder with torque_dim to use torque"
        assert torque.shape == (B, T, self.torque_dim), torque.shape
        tok = torch.cat([feat, torque], dim=-1)          # [B, T, out_dim + torque_dim]
        tokens = self.token_proj(tok)                    # [B, T, out_dim] = GPT tokens
        assert tokens.shape == (B, T, self.cfg.out_dim), tokens.shape
        return tokens


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = VisionEncoderConfig()
    enc = VisionEncoder(cfg, torque_dim=6)  # 6-dim torque (3 per joint)

    B, T = 2, 4
    frames = torch.randn(B, T, 2, cfg.in_channels, cfg.image_size, cfg.image_size)

    # Vision-only path.
    feat = enc(frames)
    assert feat.shape == (B, T, cfg.out_dim), feat.shape

    # Full per-timestep wiring with a 6-dim torque vector.
    torque = torch.randn(B, T, 6)
    tokens = enc(frames, torque)
    assert tokens.shape == (B, T, cfg.out_dim), tokens.shape

    print(f"smoke test ok | frames {tuple(frames.shape)} -> feat {tuple(feat.shape)}, tokens {tuple(tokens.shape)}")
