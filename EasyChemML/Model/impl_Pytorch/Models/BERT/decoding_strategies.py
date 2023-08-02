import numpy as np
from torch.autograd import Variable
import torch
# import time
import torch.nn.functional as F


def greedy_decod(model, src, max_len, device):
    batch_size = src.shape[0]
    outputs = np.zeros((batch_size, max_len))  # initializing the output tensor batch_size x max_len
    outputs[:, 0] = 3
    outputs = torch.LongTensor(outputs).to(device)
    src_mask = (src != 0).unsqueeze(-2)  # input mask
    tok_prob = 1

    for i in range(1, max_len):  # starting decoding loop from 1 as 0th position has start token

        trg_input = outputs[:, :i]  # updating the input to the decoder with previous step prediction
        trg_mask = np.triu(np.ones((batch_size, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
        preds = model(src, trg_input, src_mask, trg_mask)

        token_probability = F.softmax(preds, dim=-1)
        val, ix = token_probability[:, -1].data.topk(1)  # taking the token with the highest probability (greedy)
        index = ix.reshape(batch_size)  # recording the idx of the predicted token acc to the vocab

        outputs[:, i] = index  # updating the output with the tokens predicted at the present iter
        tok_prob = tok_prob * val

    return preds, outputs, tok_prob


def beam_decod_run(model, src, max_len, device, top_k=5):
    outputs = np.zeros((top_k, max_len))
    outputs[:, 0] = 3
    outputs = torch.LongTensor(outputs).to(device)

    start_token = torch.LongTensor([[3]]).to(device)           # start token
    src_mask = (src != 0).unsqueeze(-2)  # input mask
    trg_mask = np.triu(np.ones((1, 1, 1)), k=1).astype('uint8')
    trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)

    first_tok_preds = model(src, start_token, src_mask, trg_mask)

    token_probability = F.softmax(first_tok_preds, dim=-1)
    val, ix = token_probability[:, -1].data.topk(top_k)
    index = ix.reshape(top_k)
    outputs[:, 1] = index
    tok_prob = val

    last_index = np.zeros(top_k)
    last_index = torch.LongTensor(last_index).to(device)

    for i in range(2, max_len):
        beam_src = src.repeat(top_k, 1)
        beam_src_mask = (beam_src != 0).unsqueeze(-2)
        trg_input = outputs[:, :i]
        trg_mask = np.triu(np.ones((top_k, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
        preds = model(beam_src, trg_input, beam_src_mask, trg_mask)

        token_probability = F.softmax(preds, dim=-1)
        val, ix = token_probability[:, -1].data.topk(top_k)
        ix = ix.reshape(top_k * top_k)

        confidence = torch.zeros((top_k, top_k)).to(device)
        for j in range(top_k):
            confidence[j, :] = tok_prob[0][j] * val[j]
        c = confidence.reshape(top_k * top_k)
        val2, ix2 = c.topk(top_k)

        for k in range(top_k):
            if top_k == 1:
                last_index[k] = outputs[0, i - 1]
            if top_k == 3:
                if 0 <= ix2[k] < 3:
                    last_index[k] = outputs[0, i - 1]
                if 3 <= ix2[k] < 6:
                    last_index[k] = outputs[1, i - 1]
                if 6 <= ix2[k] < 9:
                    last_index[k] = outputs[2, i - 1]

            if top_k == 5:
                if 0 <= ix2[k] < 5:
                    last_index[k] = outputs[0, i - 1]
                if 5 <= ix2[k] < 10:
                    last_index[k] = outputs[1, i - 1]
                if 10 <= ix2[k] < 15:
                    last_index[k] = outputs[2, i - 1]
                if 15 <= ix2[k] < 20:
                    last_index[k] = outputs[3, i - 1]
                if 20 <= ix2[k] < 25:
                    last_index[k] = outputs[4, i - 1]

            if top_k == 25:
                if 0 <= ix2[k] < 25:
                    last_index[k] = outputs[0, i - 1]
                if 25 <= ix2[k] < 50:
                    last_index[k] = outputs[1, i - 1]
                if 50 <= ix2[k] < 75:
                    last_index[k] = outputs[2, i - 1]
                if 75 <= ix2[k] < 100:
                    last_index[k] = outputs[3, i - 1]
                if 100 <= ix2[k] < 125:
                    last_index[k] = outputs[4, i - 1]
                if 125 <= ix2[k] < 150:
                    last_index[k] = outputs[5, i - 1]
                if 150 <= ix2[k] < 175:
                    last_index[k] = outputs[6, i - 1]
                if 175 <= ix2[k] < 200:
                    last_index[k] = outputs[7, i - 1]
                if 200 <= ix2[k] < 225:
                    last_index[k] = outputs[8, i - 1]
                if 225 <= ix2[k] < 250:
                    last_index[k] = outputs[9, i - 1]
                if 250 <= ix2[k] < 275:
                    last_index[k] = outputs[10, i - 1]
                if 275 <= ix2[k] < 300:
                    last_index[k] = outputs[11, i - 1]
                if 300 <= ix2[k] < 325:
                    last_index[k] = outputs[12, i - 1]
                if 325 <= ix2[k] < 350:
                    last_index[k] = outputs[13, i - 1]
                if 350 <= ix2[k] < 375:
                    last_index[k] = outputs[14, i - 1]
                if 375 <= ix2[k] < 400:
                    last_index[k] = outputs[15, i - 1]
                if 400 <= ix2[k] < 425:
                    last_index[k] = outputs[16, i - 1]
                if 425 <= ix2[k] < 450:
                    last_index[k] = outputs[17, i - 1]
                if 450 <= ix2[k] < 475:
                    last_index[k] = outputs[18, i - 1]
                if 475 <= ix2[k] < 500:
                    last_index[k] = outputs[19, i - 1]
                if 500 <= ix2[k] < 525:
                    last_index[k] = outputs[20, i - 1]
                if 525 <= ix2[k] < 550:
                    last_index[k] = outputs[21, i - 1]
                if 550 <= ix2[k] < 575:
                    last_index[k] = outputs[22, i - 1]
                if 575 <= ix2[k] < 600:
                    last_index[k] = outputs[23, i - 1]
                if 600 <= ix2[k] < 675:
                    last_index[k] = outputs[24, i - 1]

        index = ix[ix2]
        tok_prob = val2.reshape(1, top_k)
        outputs[:, i - 1] = last_index
        outputs[:, i] = index
        # print('hello')

    return preds, outputs, tok_prob


def beam_decod_batch(model, src, max_len, device, top_k=5):
    batch_size = src.shape[0]
    outputs = np.zeros((batch_size * top_k, max_len))
    outputs[:, 0] = 3
    outputs = torch.LongTensor(outputs).to(device)

    start_token = outputs[:batch_size, :1]  # torch.LongTensor([[3]]).to(device)
    src_mask = (src != 0).unsqueeze(-2)  # input mask
    trg_mask = np.triu(np.ones((batch_size, 1, 1)), k=1).astype('uint8')
    trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)

    first_tok_preds = model(src, start_token, src_mask, trg_mask)

    token_probability = F.softmax(first_tok_preds, dim=-1)
    val, ix = token_probability[:, -1].data.topk(top_k)
    index = ix.reshape(batch_size * top_k)
    outputs[:, 1] = index
    tok_prob = val

    last_index = np.zeros(top_k)
    last_index = torch.LongTensor(last_index).to(device)

    for i in range(2, max_len):

        beam_src = src.repeat(top_k, 1)
        for ex in range(batch_size):
            start_ex = ex * top_k
            end_ex = (ex + 1) * top_k
            beam_src[start_ex:end_ex, :] = src[ex].repeat(top_k, 1)

        beam_src_mask = (beam_src != 0).unsqueeze(-2)
        trg_input = outputs[:, :i]
        trg_mask = np.triu(np.ones((batch_size * top_k, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
        preds = model(beam_src, trg_input, beam_src_mask, trg_mask)

        token_probability = F.softmax(preds, dim=-1)
        val, ix = token_probability[:, -1].data.topk(top_k)

        for exp in range(batch_size):
            start_exp = exp * top_k
            end_exp = (exp + 1) * top_k
            ix_temp = ix[start_exp:end_exp].reshape(top_k * top_k)

            confidence = torch.zeros((top_k, top_k)).to(device)
            for j in range(top_k):
                confidence[j, :] = tok_prob[exp][j] * val[start_exp + j]
            c = confidence.reshape(top_k * top_k)
            val2, ix2 = c.topk(top_k)

            for k in range(top_k):
                if top_k == 1:
                    last_index[k] = outputs[0, i - 1]
                if top_k == 3:
                    if 0 <= ix2[k] < 3:
                        last_index[k] = outputs[0, i - 1]
                    if 3 <= ix2[k] < 6:
                        last_index[k] = outputs[1, i - 1]
                    if 6 <= ix2[k] < 9:
                        last_index[k] = outputs[2, i - 1]

                if top_k == 5:
                    if 0 <= ix2[k] < 5:
                        last_index[k] = outputs[start_exp + 0, i - 1]
                    if 5 <= ix2[k] < 10:
                        last_index[k] = outputs[start_exp + 1, i - 1]
                    if 10 <= ix2[k] < 15:
                        last_index[k] = outputs[start_exp + 2, i - 1]
                    if 15 <= ix2[k] < 20:
                        last_index[k] = outputs[start_exp + 3, i - 1]
                    if 20 <= ix2[k] < 25:
                        last_index[k] = outputs[start_exp + 4, i - 1]

            index = ix_temp[ix2]
            tok_prob[start_exp:end_exp] = val2.reshape(1, top_k)
            outputs[start_exp:end_exp, i - 1] = last_index
            outputs[start_exp:end_exp, i] = index
        # print('hello')

    return preds, outputs, tok_prob
