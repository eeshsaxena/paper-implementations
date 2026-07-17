# Mamba — Selective State Space Model

Implementation of **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
> Gu, A., & Dao, T. (2023). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

## Key Idea

Standard SSMs (like S4) use **fixed** transition matrices A, B, C — they cannot selectively remember or forget based on input content.

Mamba makes B, C, and the timescale `delta` **input-dependent**, giving the model attention-like recall in **O(N) time** instead of O(N²).

```
h_t = A_bar * h_{t-1} + B_bar(x_t) * x_t    # selective state update
y_t = C(x_t) * h_t                            # selective readout
```

## Structure

```
mamba.py          # MambaConfig, SelectiveSSM, MambaBlock, Mamba model
requirements.txt  # dependencies
```

## Usage

```python
from mamba import Mamba, MambaConfig

cfg = MambaConfig(d_model=128, d_state=16, n_layers=4, vocab_size=50257)
model = Mamba(cfg)

import torch
tokens = torch.randint(0, cfg.vocab_size, (1, 64))
logits = model(tokens)            # (1, 64, 50257)
output = model.generate(tokens, max_new_tokens=32)
```

## Run

```bash
pip install -r requirements.txt
python mamba.py
```

## Paper Sections Implemented

| Section | Description | Code |
|---|---|---|
| 3.1 | Structured State Space (S4) baseline | `SelectiveSSM._selective_scan` |
| 3.2 | Selection mechanism (S6) | `SelectiveSSM.forward` |
| 3.3 | Hardware-aware algorithm | Sequential reference (see note) |
| 3.4 | Mamba Block | `MambaBlock` |
| 4 | Language model head | `Mamba` |

> Note: The paper's hardware-aware parallel scan requires custom CUDA kernels.
> This implementation uses a sequential scan for clarity and portability.
