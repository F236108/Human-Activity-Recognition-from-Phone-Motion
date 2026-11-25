"""
Task 2.6 – HARNet: the full network.

Architecture: BiLSTM → Attention → nn.Linear(2H, 6)
  - BiLSTM: input_size=9, hidden_size=64 → output (B, 128, 128)
  - Attention: 64 attention units   → context (B, 128) and alpha (B, 128)
  - Linear(128, 6)                  → 6 logits

torch.manual_seed(0) is called before creation.
Trainable parameters: 46,982  (verified by hand below).
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from layers import BiLSTM, Attention


class HARNet(nn.Module):
    """
    Human Activity Recognition Network.

    Parameter count (H=64, D=9, attn=64):
    ─────────────────────────────────────────────────────────────
    Forward LSTM cell:
      W  : 4H × D  = 256 ×  9 = 2 304
      U  : 4H × H  = 256 × 64 =16 384
      b  : 4H      =            256
      subtotal                 =18 944

    Backward LSTM cell:  same                                =18 944
    BiLSTM total                                             =37 888

    Attention:
      W_a : 64 × 2H = 64 × 128 = 8 192
      b_a : 64                   =    64
      v   : 64                   =    64
      subtotal                   = 8 320

    Linear(128 → 6):
      weight : 6 × 128 = 768
      bias   : 6        =   6
      subtotal           = 774

    Grand total = 37 888 + 8 320 + 774 = 46 982  ✓
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, input_size: int = 9, hidden_size: int = 64,
                 num_classes: int = 6, attn_size: int = 64):
        super().__init__()
        self.bilstm    = BiLSTM(input_size, hidden_size)          # (B,128,128)
        self.attention = Attention(hidden_size, attn_size)        # context (B,128), alpha (B,128)
        self.classifier = nn.Linear(2 * hidden_size, num_classes) # (B,6)

    def forward(self, x):
        """
        x : (B, 128, 9)
        returns logits (B, 6), alpha (B, 128)
        """
        H = self.bilstm(x)                       # (B, 128, 128)
        context, alpha = self.attention(H)       # (B, 128), (B, 128)
        logits = self.classifier(context)        # (B, 6)
        return logits, alpha


# ──────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    model = HARNet()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}")
    assert total_params == 46_982, f"Expected 46,982 but got {total_params}"
    print("Parameter count verified: 46,982 ✓")

    # Sanity check shapes
    B = 4
    x_dummy = torch.randn(B, 128, 9)
    logits, alpha = model(x_dummy)
    print(f"logits shape : {logits.shape}")   # (4, 6)
    print(f"alpha  shape : {alpha.shape}")    # (4, 128)

    # Verify attention sums to 1
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-5), \
        "Attention weights do not sum to 1!"
    print("Attention sums to 1 ✓")

    # Initial loss should be close to ln(6) ≈ 1.7918
    import torch.nn.functional as F
    y_dummy = torch.randint(0, 6, (B,))
    loss = F.cross_entropy(logits, y_dummy)
    print(f"Initial loss sample: {loss.item():.4f}  (reference ln 6 ≈ 1.7918)")
