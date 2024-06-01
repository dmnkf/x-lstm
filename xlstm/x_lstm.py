from itertools import repeat
from typing import List, Tuple, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim import Optimizer
from torchmetrics import Accuracy, F1Score

from xlstm import mLSTM, sLSTM
from xlstm.utils import Hidden

OptimizerCallable = Callable[[Iterable], Optimizer]

class xLSTM(LightningModule):
    '''The extended Long Short Term Memory (xLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    This model stacks sLSTM and mLSTM modules with residual
    connections and offers superior memory and performance
    compared to the standard LSTM model, achieving competitive
    or better performance and scaling than Transformer models
    or State-Space models.

    DISCLAIMER:
    This code was heavily inspired by already existing implementations of the xLSTM model.

    All the text embedding specific details were removed and adjusted accordingly to work with any sequential input.

    The original repositories can be found here:
    - https://github.com/muditbhargava66/PyxLSTM
    - https://github.com/myscience/x-lstm

    '''

    def __init__(
            self,
            num_layers : int,
            signature : Tuple[int, int],
            inp_dim : int,
            head_dim : int,
            head_num : int,
            out_dim : int = None,
            p_factor : Tuple[float, float] = (2, 4/3),
            ker_size : int = 4,
            optimizer : OptimizerCallable = AdamW,
            only_last: bool = False,
    ) -> None:
        '''Initialize the LLM model.

        Args:
            num_layers (int): The number of layers in the model.
            signature (Tuple[int, int]): The signature of the model,
                which represents the ratio of the mLSTM-to-sLSTM blocks.
            inp_dim (int): The dimension of the input sequence.
            head_dim (int): The dimension of each attention head.
            head_num (int): The number of attention heads.
            out_dim (int): The dimension of the output logits.
            p_factor (Tuple[float, float], optional): The expansion factor
                for the MLP projection in the m|s-LSTM blocks. Defaults to (2, 4/3).
            ker_size (int, optional): The kernel size for the causal convolutional layers.
                Defaults to 4.
            only_last (bool, optional): Whether to return only the last sequence output of the model. (e.g. for classification).

        '''
        super().__init__()

        self.accuracy = Accuracy(num_classes=out_dim, task='multiclass')
        self.f1_score = F1Score(num_classes=out_dim, average='weighted', task='multiclass')
        self.optimizer = optimizer
        self.only_last = only_last
        if out_dim is None:
           out_dim = inp_dim


        m_factor, s_factor = p_factor

        mlstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : m_factor,
            'ker_size' : ker_size,
        }

        slstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : s_factor,
            'ker_size' : ker_size,
        }

        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num

        self.model : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w, _ in zip(repeat(which), range(num_layers))
        ])

        self.head = nn.Linear(inp_dim, out_dim, bias=False)

        self.save_hyperparameters()

    def forward(
            self,
            seq: Tensor,
            hid: Hidden | None = None,
            batch_first : bool = True,
    ) -> Tuple[Tensor, Hidden]:
        '''Forward pass of the xLSTM model.

        Args:
            seq (Tensor): Input tensor representing the sequence.
                Expected shape: (batch, seq_len) if batch_first=True,
                else (seq_len, batch).
            hid (Hidden, optional): Cache object for storing intermediate hidden
                values of the m|s-LSTM blocks of the model. If None, the hidden
                states are initialized by the models. Defaults to None.

        Returns:
            Tuple[Tensor, Hidden]: Returns tensor of predicted logits of shape
                (batch, seq_len, input_dim (features)) if batch_first=True or of shape
                (seq_len, batch, input_dim (features)) if batch_first=False, and the
                updated hidden model states.
        '''


        if batch_first: seq = rearrange(seq, 'b s i -> s b i')
        if hid is None: hid = [l.init_hidden(seq.size(1)) for l in self.model]

        # Pass the sequence through the mLSTM and sLSTM blocks
        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.model):
                inp, hid[i] = lstm(inp, hid[i])

            out.append(inp)

        out = torch.stack(out, dim=1 if batch_first else 0)
        out = self.head(out)
        if self.only_last:
            out = out[:, -1, :] if batch_first else out[-1, :, :]

        return out, hid

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits, hid = self(x)
        loss = F.cross_entropy(logits, y.float())

        y_index = torch.argmax(y, dim=1).float()
        preds = torch.argmax(logits, dim=1).float()
        # Some weird behaviour with torchmetrics when calculating the metrics with the logits and the one-hot encoded labels
        acc = self.accuracy(preds, y_index)
        f1 = self.f1_score(preds, y_index)
        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "train_acc": acc, "train_f1": f1}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._shared_step(batch, batch_idx)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )

        return optim
