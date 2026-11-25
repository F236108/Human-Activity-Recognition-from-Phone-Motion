"""
Task 2 – Recurrent layers written from scratch in PyTorch.

Contains:
  RNNCell   – plain tanh RNN (not used in final network; kept for Task 6.6 comparison)
  LSTMCell  – 4-gate LSTM cell, all gates packed in one matrix
  LSTM      – runs LSTMCell over T time steps, returns all hidden states
  BiLSTM    – two LSTM layers, forward + reversed, concatenated per time step
  Attention – additive (Bahdanau-style) attention over time; returns context + weights

Allowed: nn.Module, nn.Parameter, nn.Linear, torch.sigmoid, torch.tanh,
         torch.softmax, torch.optim.Adam, torch.nn.utils.clip_grad_norm_,
         numpy, matplotlib.
Not allowed: nn.RNN, nn.LSTM, nn.GRU, nn.*Cell, nn.MultiheadAttention,
             F.scaled_dot_product_attention, scikit-learn, Keras, TensorFlow.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: uniform init in [-1/sqrt(H), 1/sqrt(H)]
# ---------------------------------------------------------------------------
def _uniform_init(tensor, H):
    bound = 1.0 / math.sqrt(H)
    nn.init.uniform_(tensor, -bound, bound)


# ---------------------------------------------------------------------------
# 2.1  RNN Cell
# ---------------------------------------------------------------------------
class RNNCell(nn.Module):
    """
    h_t = tanh(W x_t + U h_{t-1} + b)

    Parameters
      W : (H, D)
      U : (H, H)
      b : (H,)
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H, D = hidden_size, input_size
        self.H = H

        self.W = nn.Parameter(torch.empty(H, D))
        self.U = nn.Parameter(torch.empty(H, H))
        self.b = nn.Parameter(torch.zeros(H))

        _uniform_init(self.W, H)
        _uniform_init(self.U, H)
        _uniform_init(self.b, H)

    def forward(self, x, h_prev):
        """
        x      : (B, D)
        h_prev : (B, H)
        returns h_t : (B, H)
        """
        return torch.tanh(x @ self.W.t() + h_prev @ self.U.t() + self.b)


# ---------------------------------------------------------------------------
# 2.2  LSTM Cell
# ---------------------------------------------------------------------------
class LSTMCell(nn.Module):
    """
    All four gates packed in a single weight matrix, stacked as:
      [input gate i, forget gate f, cell candidate g, output gate o]

    W : (4H, D),  U : (4H, H),  b : (4H)

    Gates (after splitting the linear combination into 4 chunks of size H):
      i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
      f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
      g_t = tanh   (W_g x_t + U_g h_{t-1} + b_g)
      o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      h_t = o_t ⊙ tanh(c_t)
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H, D = hidden_size, input_size
        self.H = H

        self.W = nn.Parameter(torch.empty(4 * H, D))
        self.U = nn.Parameter(torch.empty(4 * H, H))
        self.b = nn.Parameter(torch.zeros(4 * H))

        _uniform_init(self.W, H)
        _uniform_init(self.U, H)
        _uniform_init(self.b, H)

    def forward(self, x, h_prev, c_prev):
        """
        x      : (B, D)
        h_prev : (B, H)
        c_prev : (B, H)
        returns h_t, c_t  each (B, H)
        """
        gates = x @ self.W.t() + h_prev @ self.U.t() + self.b   # (B, 4H)
        H = self.H
        i = torch.sigmoid(gates[:, 0*H : 1*H])   # input gate
        f = torch.sigmoid(gates[:, 1*H : 2*H])   # forget gate
        g = torch.tanh   (gates[:, 2*H : 3*H])   # cell candidate
        o = torch.sigmoid(gates[:, 3*H : 4*H])   # output gate

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ---------------------------------------------------------------------------
# 2.3  LSTM (layer)
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    """
    Runs LSTMCell over T = 128 time steps.
    Returns all hidden states: (B, T, H).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.H    = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        """
        x : (B, T, D)
        returns hidden_states : (B, T, H)
        """
        B, T, _ = x.shape
        H = self.H
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)

        all_h = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
            all_h.append(h)                       # each (B, H)

        return torch.stack(all_h, dim=1)          # (B, T, H)


# ---------------------------------------------------------------------------
# 2.4  Bidirectional LSTM
# ---------------------------------------------------------------------------
class BiLSTM(nn.Module):
    """
    Two separate LSTM layers (different weights):
      - forward  LSTM reads x_1 ... x_T
      - backward LSTM reads x_T ... x_1 (reversed in time),
        then its outputs are reversed back so position t aligns with x_t

    Concatenates at every time step: h_t = [h_forward_t ; h_backward_t]
    Output shape: (B, T, 2H)
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.forward_lstm  = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):
        """
        x : (B, T, D)
        returns : (B, T, 2H)
        """
        h_fwd = self.forward_lstm(x)                        # (B, T, H)
        # Reverse time dimension for backward pass
        x_rev = torch.flip(x, dims=[1])                    # (B, T, D)
        h_bwd = self.backward_lstm(x_rev)                  # (B, T, H)
        # Reverse outputs back so position t aligns with x_t
        h_bwd = torch.flip(h_bwd, dims=[1])                # (B, T, H)

        return torch.cat([h_fwd, h_bwd], dim=-1)           # (B, T, 2H)


# ---------------------------------------------------------------------------
# 2.5  Attention over time
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    """
    Additive attention over T time steps.

    Parameters:
      W_a : (attn_size, 2H)   = (64, 128)
      b_a : (attn_size,)
      v   : (attn_size,)      = (64,)

    Scores:
      e_t = v^T tanh(W_a h_t + b_a)
    Weights:
      alpha_t = softmax(e_t)             shape (B, T)
    Context:
      c = sum_t alpha_t h_t              shape (B, 2H)

    Returns c (B, 2H) and alpha (B, T).
    """
    def __init__(self, hidden_size: int, attn_size: int = 64):
        super().__init__()
        input_dim = 2 * hidden_size    # BiLSTM output is 2H
        self.W_a = nn.Parameter(torch.empty(attn_size, input_dim))
        self.b_a = nn.Parameter(torch.zeros(attn_size))
        self.v   = nn.Parameter(torch.empty(attn_size))

        _uniform_init(self.W_a, hidden_size)
        _uniform_init(self.b_a, hidden_size)
        _uniform_init(self.v,   hidden_size)

    def forward(self, H):
        """
        H : (B, T, 2H)
        returns context (B, 2H), alpha (B, T)
        """
        # e_t = v^T tanh(W_a h_t + b_a)
        # H @ W_a.t() -> (B, T, attn_size);  + b_a broadcasts
        scores = torch.tanh(H @ self.W_a.t() + self.b_a)   # (B, T, attn_size)
        e = scores @ self.v                                  # (B, T)

        alpha = torch.softmax(e, dim=-1)                     # (B, T), sums to 1

        # Weighted sum: context = sum_t alpha_t h_t
        # alpha: (B, T) -> (B, T, 1) for broadcasting
        context = (alpha.unsqueeze(-1) * H).sum(dim=1)      # (B, 2H)

        return context, alpha
