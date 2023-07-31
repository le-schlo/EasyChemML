import numpy as np, torch
import torch.nn.functional
from torch import nn
from torch import device
from torch.optim import Adam
from torch.autograd import Variable
from EasyChemML.Model.impl_Pytorch.Models.BERT.__BERT_Transformer_nn import BERT_Transformer_nn


class FP2MOL_Bert:
    __model: BERT_Transformer_nn
    __device: device
    __optimiser: Adam

    def __init__(self, src_vocab_size: int = 1024, trg_vocab_size: int = 105, N_coderLayers: int = 6,
                 att_heads: int = 8, d_model: int = 512, dropout: float = 0.1, max_seq_len: int = 100, torch_device: str = 'cuda:0'):
        self.__model = BERT_Transformer_nn(src_vocab_size, trg_vocab_size, d_model, N_coderLayers, att_heads, dropout, max_seq_len)
        self.__device = torch.device(torch_device)
        self.__model.to(self.__device)
        self.__optimiser = torch.optim.Adam(self.__model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def init_internal_parameters(self):
        for p in self.__model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_masks(self, src, trg_input):
        input_seq = src
        # creates mask with 0s wherever there is padding in the input
        input_msk = (input_seq != 0).unsqueeze(-2).to(self.__device)

        target_seq = trg_input
        target_msk = (target_seq != 0).unsqueeze(1)
        size = target_seq.size(1)  # get seq_len for matrix
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(self.__device)

        target_msk = target_msk & nopeak_mask

        return input_msk, target_msk

    def fit_CalcLoss(self, x: np.ndarray, y: np.ndarray):
        self.__model.train()
        # total_loss = 0

        x = torch.LongTensor(x).to(self.__device)
        y = torch.LongTensor(y).to(self.__device)

        trg_input = y[:, :-1]
        targets = y[:, 1:].contiguous().view(-1)

        src_mask, trg_mask = self.create_masks(x, trg_input)
        preds = self.__model(x, trg_input, src_mask, trg_mask)

        self.__optimiser.zero_grad()

        loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), targets)
        loss.backward()
        self.__optimiser.step()

        # total_loss += torch.Tensor.item(loss.data)
        return loss

    def get_model(self):
        return self.__model

    def get_optimiser(self):
        return self.__optimiser

    def fit_eval(self):
        pass

    def save_model(self, p_fname, o_fname):
        torch.save(self.__model.state_dict(), p_fname)
        torch.save(self.__optimiser.state_dict(), o_fname)

