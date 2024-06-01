# FORK NOTE
This fork removes all embedding specific code and generalizes the models to work more seamlessly with any sequential input. It also introduces a dedicated output dimension for the xLSTM model.

As described in the paper, the goal was to challenge transformers, hence the models were meant to be used with some kind of embedding.

Anyway, as the core of the xLSTM are still LSTM cells, any sequential input should work. 

If this really makes sense is another question, but I wanted to try it out.

# xLSTM barebone in PyTorch Lightning

This repo contains the _unofficial_ implementation of `xLSTM` model as introduced in [Beck et al. (2024)](https://arxiv.org/abs/2405.04517). This repo is developed mainly for didactic purposes to spell out the details of a modern `Long-Short Term Memory` with competitive performances against modern `Transformers` or `State-Space` models (e.g. `Mamba`).


## Usage

To train the model, simply run the following command:

```python
from xlstm import xLSTM
import torch

batch_size = 32
seq_len = 100
inp_dim = 16

xlstm = xLSTM(
            num_layers = 2,
            signature = (7, 1),
            inp_dim= inp_dim,
            head_dim= 8,
            head_num= 4,
            out_dim= 24,
            p_factor= (2, 4/3),
            ker_size = 4,
            only_last = False
        )


seq = torch.randn(batch_size, seq_len, inp_dim)
out = xlstm(seq)
``` 

# Requirements

Code was tested with Python 3.11+. To install the required dependencies simply run `pip install -r requirements.txt`.

```
torch==2.3.0
PyYAML==6.0.1
einops==0.8.0
lightning==2.2.4
setuptools==69.5.1
```

# Citations

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
