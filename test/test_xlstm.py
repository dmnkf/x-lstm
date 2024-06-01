import unittest

import torch
from os import path

from xlstm import xLSTM

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

class TestXLSTM(unittest.TestCase):
    def setUp(self):
        self.num_layers = 8
        self.signature = (7, 1)
        self.inp_dim = 16
        self.head_dim = 8
        self.head_num = 4
        self.ker_size = 4
        self.p_factor = (2, 4/3)
        self.output_dim = 24

        self.seq_len = 32
        self.batch_size = 4

        # Mockup input for example purposes
        self.seq = torch.randn(self.batch_size, self.seq_len, self.inp_dim)

    def test_forward(self):
        
        xlstm = xLSTM(
            num_layers = self.num_layers,
            signature = self.signature,
            inp_dim= self.inp_dim,
            head_dim= self.head_dim,
            head_num= self.head_num,
            out_dim= self.output_dim,
            p_factor= self.p_factor,
            ker_size = self.ker_size,
            only_last = False
        )

        # Compute the output using the xLSTM architecture
        out, _ = xlstm.forward(self.seq, batch_first=True)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.output_dim))
            

if __name__ == '__main__':
    unittest.main()