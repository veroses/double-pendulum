from dataclasses import dataclass, field
from encoders import *
from transformer import GPT, GPTConfig

@dataclass
class DynamicsModelConfig:
    encoder: MLPEncoderConfig | ViTEncoderConfig
    dynamics: GPTConfig
    head_hidden: list[int] = field(default_factory=list)
    history_length: int = 10
    normalize_input: bool = True
    normalize_output: bool = True
   


