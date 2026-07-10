"""Causal-transformer dynamics model for the double pendulum.

The model is a thin, NeRD-style wrapper around three pieces:

    raw input  ->  [ input encoder ]  ->  [ GPT dynamics core ]  ->  [ Δs head ]

  1. Input encoder  — turns one timestep of observation into a token. There are
     two interchangeable encoders, chosen by the *type of the encoder config*
     (this is the "switch"):
        * MLPEncoderConfig   -> MLPEncoder    (proprioceptive state vector)
        * VisionEncoderConfig -> VisionEncoder (two camera views + torque)
  2. GPT dynamics core — a causal GPT-2-style transformer that attends over the
     history window and mixes the per-step tokens (see transformer.py).
  3. Δs head — a small MLP (a reused MLPEncoder) that maps the core output to
     the predicted state delta. The next state is recovered elsewhere as
     s_{t+1} = s_t ⊕ Δs.
"""

from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn

from encoders import (
    MLPEncoder,
    MLPEncoderConfig,
    VisionEncoder,
    VisionEncoderConfig,
)
from transformer import GPT, GPTConfig


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DynamicsModelConfig:
    # The encoder config's *type* is the modality switch: an MLPEncoderConfig
    # selects the state encoder, a VisionEncoderConfig selects the vision one.
    encoder: MLPEncoderConfig | VisionEncoderConfig
    # GPT template. `in_features` is overwritten at construction time with the
    # encoder's output width, so its value here is ignored.
    dynamics: GPTConfig
    # Δs head. It is just another MLPEncoder over the GPT output, so it reuses
    # the same config type. `layer_sizes` here are the head's HIDDEN layers; the
    # final projection to the (data-driven) Δs width is appended automatically
    # at construction, so an empty config gives a single Linear.
    head: MLPEncoderConfig = field(default_factory=MLPEncoderConfig)
    history_length: int = 10
    normalize_input: bool = True
    normalize_output: bool = True


# --------------------------------------------------------------------------- #
# Dynamics model
# --------------------------------------------------------------------------- #
# Per-modality names of the tensors expected in the input dict. Keeping them as
# constants makes the (de)normalisation bookkeeping below explicit.
STATE_KEY = "state"      # MLP path: the full proprioceptive state vector
FRAMES_KEY = "frames"    # vision path: [B, T, 2, 3, H, W]
TORQUE_KEY = "torque"    # vision path: [B, T, torque_dim]


class DynamicsModel(nn.Module):
    """Encoder → GPT → Δs head, with optional running-stat (de)normalisation.

    Args:
        config:     DynamicsModelConfig (also carries the modality switch).
        output_dim: width of the predicted state delta Δs.
        input_size: per-timestep state width — required for the MLP path,
                    ignored for the vision path.
        torque_dim: torque width — required for the vision path (torque is
                    concatenated onto the visual feature), ignored for MLP.
        device:     torch device the module lives on.
    """

    def __init__(
        self,
        config: DynamicsModelConfig,
        output_dim: int,
        input_size: int | None = None,
        torque_dim: int | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.config = config

        # Running mean/std stats, filled in later via set_input_rms/​set_output_rms.
        # Normalisation is a no-op until the corresponding stats are provided, so
        # the model is runnable before they have been fitted.
        self.input_rms = None
        self.output_rms = None
        self.normalize_input = config.normalize_input
        self.normalize_output = config.normalize_output

        # --- modality switch: build the encoder and learn its output width --- #
        self.encoder, self.feature_dim, self.modality = self._build_encoder(
            config.encoder, input_size=input_size, torque_dim=torque_dim
        )

        # --- GPT dynamics core: its input width tracks the encoder's output --- #
        gpt_cfg = replace(config.dynamics, in_features=self.feature_dim)
        assert config.history_length <= gpt_cfg.block_size, (
            f"history_length={config.history_length} exceeds GPT "
            f"block_size={gpt_cfg.block_size}"
        )
        self.dynamics = GPT(gpt_cfg)
        core_dim = gpt_cfg.n_embd

        # --- Δs head --- #
        # The head is a plain MLP, which is exactly what MLPEncoder already is,
        # so we reuse it rather than duplicate the Linear-stack logic. The head
        # config carries only the hidden layers; the final projection to the
        # data-driven Δs width is appended here so it can never be mis-set.
        head_cfg = replace(
            config.head, layer_sizes=[*config.head.layer_sizes, output_dim]
        )
        self.head = MLPEncoder(core_dim, head_cfg)
        assert self.head.output_dim == output_dim, self.head.output_dim
        self.output_dim = output_dim

        self.to(device)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_encoder(self, encoder_cfg, input_size, torque_dim):
        """Dispatch on the encoder config type and return (encoder, dim, modality)."""
        if isinstance(encoder_cfg, VisionEncoderConfig):
            assert torque_dim is not None, "vision path requires torque_dim"
            encoder = VisionEncoder(encoder_cfg, torque_dim=torque_dim)
            return encoder, encoder_cfg.out_dim, "vision"

        if isinstance(encoder_cfg, MLPEncoderConfig):
            assert input_size is not None, "MLP path requires input_size"
            encoder = MLPEncoder(input_size, encoder_cfg)
            return encoder, encoder.output_dim, "mlp"

        raise NotImplementedError(f"unsupported encoder config: {type(encoder_cfg)!r}")

    # ------------------------------------------------------------------ #
    # Running-stat normalisation (optional)
    # ------------------------------------------------------------------ #
    def set_input_rms(self, data_rms):
        """Register input running-mean/std objects, keyed by input-dict name."""
        self.input_rms = {k: data_rms[k] for k in self._input_keys() if k in data_rms}

    def set_output_rms(self, output_rms):
        self.output_rms = output_rms

    def _input_keys(self):
        return (STATE_KEY,) if self.modality == "mlp" else (FRAMES_KEY, TORQUE_KEY)

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def _extract_features(self, input_dict):
        """Encode one input dict into the per-step token sequence [B, T, feature_dim]."""
        if self.modality == "mlp":
            return self.encoder(input_dict[STATE_KEY])
        # vision: two camera views per step + the (unseen-in-pixels) torque.
        return self.encoder(input_dict[FRAMES_KEY], input_dict[TORQUE_KEY])

    def _maybe_normalize_input(self, input_dict):
        if not (self.normalize_input and self.input_rms):
            return input_dict
        # Copy so we never mutate the caller's tensors in place.
        out = dict(input_dict)
        for key, rms in self.input_rms.items():
            out[key] = rms.normalize(out[key])
        return out

    def _maybe_unnormalize_output(self, output):
        if self.normalize_output and self.output_rms is not None:
            return self.output_rms.normalize(output, un_norm=True)
        return output

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #
    def forward(self, input_dict, inject_noise=False):
        """Multi-step forward over a history window.

        Inputs are shaped [B, T, ...]; returns Δs of shape [B, T, output_dim]
        (one prediction per timestep, as needed for sequence training).
        """
        input_dict = self._maybe_normalize_input(input_dict)

        if inject_noise:
            # Small Gaussian input perturbation — a NeRD-style robustness aug
            # that keeps autoregressive rollouts from drifting off-distribution.
            input_dict = {
                k: v + torch.randn_like(v) * 0.01 for k, v in input_dict.items()
            }

        features = self._extract_features(input_dict)   # [B, T, feature_dim]
        core = self.dynamics(features)                  # [B, T, n_embd]
        output = self.head(core)                        # [B, T, output_dim]
        return self._maybe_unnormalize_output(output)

    def evaluate(self, input_dict):
        """Single-step prediction: run the window but keep only the last Δs.

        Returns shape [B, 1, output_dim] — convenient for autoregressive rollout
        where only the newest prediction is consumed.
        """
        output = self.forward(input_dict)
        return output[:, -1:, :]
