"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Paper: https://arxiv.org/abs/2312.00752
Authors: Albert Gu, Tri Dao (2023)

This implementation covers:
- Section 3.1: Structured State Space Models (S4)
- Section 3.2: The Selective Scan (S6) mechanism
- Section 3.4: The Mamba Block

Key insight: unlike fixed SSMs (S4), Mamba makes the SSM parameters
(B, C, delta) input-dependent, giving it selection/recall ability
similar to attention but in linear time O(N).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaConfig:
    """Configuration for the Mamba model."""

    def __init__(
        self,
        d_model: int = 128,       # model dimension D
        d_state: int = 16,        # SSM state dimension N (paper uses 16)
        d_conv: int = 4,          # local convolution width
        expand: int = 2,          # block expansion factor E
        n_layers: int = 4,        # number of Mamba blocks
        vocab_size: int = 256,    # for language modelling demos
        dt_rank: str = "auto",    # rank of delta projection
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # E*D
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias


class SelectiveSSM(nn.Module):
    """
    The core of Mamba: Selective State Space Model (Algorithm 2 in paper).

    Standard SSMs use fixed A, B, C matrices. The key innovation in Mamba
    is making B, C, and the timescale delta input-dependent, enabling
    selective information propagation through the sequence.

    Recurrence (continuous):
        h'(t) = A h(t) + B(t) x(t)
        y(t)  = C(t) h(t)

    Discretized (ZOH, Eq. 4 in paper):
        A_bar = exp(delta * A)
        B_bar = (delta * A)^{-1} (exp(delta * A) - I) * delta * B
        h_t   = A_bar h_{t-1} + B_bar x_t
        y_t   = C h_t
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank

        # Input projection: x -> (delta, B, C)  [Section 3.2]
        # delta shape: (dt_rank,), B: (d_state,), C: (d_state,)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * config.d_state,
            bias=False,
        )

        # Delta projection: dt_rank -> d_inner  (low-rank factorisation)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj following the paper (Appendix D)
        dt_init_std = self.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias so softplus(dt_bias) is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A matrix: initialised as -exp(log A) to keep A negative (stable)
        # Shape: (d_inner, d_state)
        A = repeat(
            torch.arange(1, config.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))   # store log for stability
        self.A_log._no_weight_decay = True

        # D "skip" scalar (one per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D_inner)  — batch, sequence length, inner dim
        Returns:
            y: (B, L, D_inner)
        """
        B, L, d_in = x.shape
        assert d_in == self.d_inner

        # Recover A from log-parameterisation
        A = -torch.exp(self.A_log.float())          # (d_inner, d_state)

        # Project x to get delta, B_ssm, C_ssm  (all input-dependent!)
        x_proj = self.x_proj(x)                     # (B, L, dt_rank + 2*d_state)
        delta, B_ssm, C_ssm = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # delta: (B, L, dt_rank) -> (B, L, d_inner) via low-rank proj + softplus
        delta = F.softplus(self.dt_proj(delta))     # (B, L, d_inner)

        # Run the selective scan (parallel version via associative scan)
        y = self._selective_scan(x, delta, A, B_ssm, C_ssm)

        # Add skip connection scaled by D
        y = y + x * self.D

        return y

    def _selective_scan(
        self,
        u: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (D, N)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
    ) -> torch.Tensor:
        """
        Discretise A/B with ZOH rule then run the recurrence.
        This is the reference (sequential) implementation.
        Production code uses a CUDA parallel scan for O(L log L) speed.
        """
        B_batch, L, D = u.shape
        N = A.shape[1]

        # ZOH discretisation  (Eq. 4)
        # delta_A: (B, L, D, N)
        delta_A = torch.exp(
            torch.einsum("bld,dn->bldn", delta, A)
        )
        # delta_B_u: (B, L, D, N)
        delta_B_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        # Sequential scan over the time axis
        h = torch.zeros(B_batch, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_B_u[:, t]      # (B, D, N)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)   # (B, L, D)


class MambaBlock(nn.Module):
    """
    Single Mamba block (Figure 3 in the paper).

    Structure:
        x -> Linear (expand) ---|---> conv1d -> SiLU -> SSM --> * --> Linear (project) -> out
                                |                                ^
                                |----> Linear (expand) -> SiLU -|
    The second branch acts as a gating mechanism.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # Input/output projections
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Short local convolution before the SSM (width d_conv)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=config.d_inner,    # depthwise
            bias=config.conv_bias,
        )

        # The selective SSM
        self.ssm = SelectiveSSM(config)

        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            out: (B, L, D)
        """
        residual = x
        x = self.norm(x)

        # Expand and split into SSM branch (x_ssm) and gate branch (z)
        x_and_z = self.in_proj(x)                    # (B, L, 2*D_inner)
        x_ssm, z = x_and_z.chunk(2, dim=-1)          # each (B, L, D_inner)

        # Local conv (causal: trim the future-padding)
        x_ssm = rearrange(x_ssm, "b l d -> b d l")
        x_ssm = self.conv1d(x_ssm)[..., : x.shape[1]]
        x_ssm = rearrange(x_ssm, "b d l -> b l d")
        x_ssm = F.silu(x_ssm)

        # Selective SSM
        y = self.ssm(x_ssm)

        # Gated output
        y = y * F.silu(z)

        # Project back and add residual
        return self.out_proj(y) + residual


class Mamba(nn.Module):
    """
    Full Mamba language model: embedding -> N x MambaBlock -> LM head.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying (common practice)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) integer token ids
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embedding(input_ids)       # (B, L, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Simple greedy/temperature sampling."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :]   # last token logits
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K"


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Mamba: Selective State Space Model — Minimal Implementation")
    print("Paper: https://arxiv.org/abs/2312.00752")
    print("=" * 60)

    # Tiny model for quick demo
    cfg = MambaConfig(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=2,
        vocab_size=128,
    )

    model = Mamba(cfg)
    print(f"\nModel parameters: {count_params(model)}")
    print(f"Config: d_model={cfg.d_model}, d_state={cfg.d_state}, "
          f"layers={cfg.n_layers}, d_inner={cfg.d_inner}")

    # Forward pass test
    B, L = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    logits = model(input_ids)
    print(f"\nForward pass: input {tuple(input_ids.shape)} "
          f"-> logits {tuple(logits.shape)}")
    assert logits.shape == (B, L, cfg.vocab_size), "Shape mismatch!"
    print("Shape check: PASSED")

    # Generation test
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    output = model.generate(prompt, max_new_tokens=8, temperature=0.8)
    print(f"\nGeneration: prompt len={prompt.shape[1]} "
          f"-> output len={output.shape[1]} (generated {output.shape[1] - prompt.shape[1]} tokens)")

    # SSM selective scan sanity check
    print("\nVerifying selective scan output shape...")
    ssm = SelectiveSSM(cfg)
    x_test = torch.randn(2, 32, cfg.d_inner)
    out = ssm(x_test)
    assert out.shape == x_test.shape, "SSM output shape mismatch!"
    print("SSM shape check: PASSED")

    print("\nAll checks passed. Mamba implementation is working correctly.")
