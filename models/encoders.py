from dataclasses import dataclass, field

@dataclass
class MLPEncoderConfig:
    layer_sizes: list[int] = field(default_factory=list)  # [] = identity
    activation: str = "relu"

@dataclass
class ViTEncoderConfig:
    image_size: int = 64
    patch_size: int = 8
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.0
    use_spt: bool = True
    use_lsa: bool = True

class MLPEncoder(nn.Module):
    pass

class ViTEncoder(nn.Module):
    pass