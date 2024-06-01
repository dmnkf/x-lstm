from math import sqrt
from typing import Tuple

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor
from torch import exp
from torch import sigmoid
from torch.nn.functional import silu

from xlstm.utils import CausalConv1d, enlarge_as


class mLSTM(nn.Module):
    '''The matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers superior memory due to its storing values in a
    matrix instead of a scalar. It is fully parallelizable
    and updates internal memory with the covariance rule.
    '''

    def __init__(
            self,
            inp_dim : int,
            head_num : int,
            head_dim : int,
            p_factor : int = 2,
            ker_size : int = 4,
    ) -> None:
        super().__init__()

        self.inp_dim = inp_dim
        self.head_num = head_num
        self.head_dim = head_dim

        hid_dim = head_num * head_dim

        self.inp_norm = nn.LayerNorm(inp_dim)
        self.hid_norm = nn.GroupNorm(head_num, hid_dim)

        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        self.up_l_proj = nn.Linear(inp_dim, int(p_factor * inp_dim))
        self.up_r_proj = nn.Linear(inp_dim, hid_dim)
        self.down_proj = nn.Linear(hid_dim, inp_dim)

        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)

        self.skip = nn.Conv1d(int(p_factor * inp_dim), hid_dim, kernel_size=1, bias=False)

        self.W_i = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_f = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_o = nn.Linear(int(p_factor * inp_dim), hid_dim)

        self.W_q = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_k = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_v = nn.Linear(int(p_factor * inp_dim), hid_dim)

    @property
    def device(self) -> str:
        '''Get the device of the model.

        Returns:
            str: The device of the model.
        '''
        return next(self.parameters()).device

    def init_hidden(self, bs : int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Initialize the hidden state of the sLSTM model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        '''

        c_0 = torch.zeros(bs, self.head_num, self.head_dim, self.head_dim, device=self.device)
        n_0 = torch.ones (bs, self.head_num, self.head_dim               , device=self.device)
        m_0 = torch.zeros(bs, self.head_num                              , device=self.device)

        return c_0, n_0, m_0

    def forward(
            self,
            seq: Tensor,
            hid: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        '''_summary_

        Args:
            seq (Tensor): _description_
            hid (Tuple[Tensor, Tensor]): _description_

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: _description_
        '''

        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, m_tm1 = hid

        x_n : Tensor = self.inp_norm(seq) # shape: b i

        x_t = self.up_l_proj(x_n) # shape: b (i * p_factor)
        r_t = self.up_r_proj(x_n) # shape: b (h d)

        # Compute the causal convolutional input (to be 
        # used for the query and key gates)
        x_c = self.causal_conv(x_t) # shape: b 1 (i * p_factor)
        x_c = silu(x_c).squeeze()   # shape: b (i * p_factor)

        q_t = rearrange(self.W_q(x_c), 'b (h d) -> b h d', h=self.head_num)
        k_t = rearrange(self.W_k(x_c), 'b (h d) -> b h d', h=self.head_num) / sqrt(self.head_dim)
        v_t = rearrange(self.W_v(x_t), 'b (h d) -> b h d', h=self.head_num)

        i_t: Tensor = self.W_i(x_c) # shape: b h
        f_t: Tensor = self.W_f(x_c) # shape: b h
        o_t: Tensor = self.W_o(x_t) # shape: b (h d)

        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)

        i_t = exp(i_t - m_t)         # Eq. (25) in ref. paper
        f_t = exp(f_t - m_t + m_tm1) # Eq. (26) in ref. paper
        o_t = sigmoid(o_t)           # Eq. (27) in ref. paper

        # Update the internal states of the model
        c_t = enlarge_as(f_t, c_tm1) * c_tm1 + enlarge_as(i_t, c_tm1) * einsum(v_t, k_t, 'b h d, b h p -> b h d p')
        n_t = enlarge_as(f_t, n_tm1) * n_tm1 + enlarge_as(i_t, k_t)   * k_t
        h_t = o_t * rearrange(
            einsum(c_t, q_t, 'b h d p, b h p -> b h d') /
            einsum(n_t, q_t, 'b h d, b h d -> b h').clamp(min=1).unsqueeze(-1),
            'b h d -> b (h d)'
        ) # Eq. (21) in ref. paper

        x_c = rearrange(x_c, 'b i -> b i 1')
        out = self.hid_norm(h_t) + self.skip(x_c).squeeze() # shape: b (h d)
        out = out * silu(r_t)                               # shape: b (h d)
        out = self.down_proj(out)                           # shape: h i

        # Return output with the residual connection and the
        # newly updated hidden state.
        return out + seq, (c_t, n_t, m_t)