import torch
import torch.nn as nn
from torch.autograd import Variable

from torchcrf import CRF

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, pading_idx, hidden_size, n_layers, dropout, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size  = input_size
        self.emb_size    = emb_size
        self.pading_idx  = pading_idx
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        # self.dropout     = dropout

        self.embedding   = nn.Embedding(input_size, emb_size, padding_idx=pading_idx)
        self.lstm        = nn.LSTM(input_size=emb_size,
                                   hidden_size=hidden_size,
                                   bidirectional=bidirectional,
                                   batch_first=True)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, seqIn):
        embedded = self.dropout(self.embedding(seqIn))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden


class AttnIntent(nn.Module):
    def __init__(self):
        super(AttnIntent, self).__init__()

    def forward(self, hidden, outputs, mask=None):
        hidden = hidden.squeeze(1).unsqueeze(2)

        batch_size    = outputs.size(0)
        max_len       = outputs.size(1)

        energies      = outputs.contiguous().view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.squeeze(1).masked_fill(mask, -1e12)

        alpha         = torch.softmax(attn_energies, dim=-1)
        alpha         = alpha.unsqueeze(1)
        context       = alpha.bmm(outputs)
        context       = context.squeeze(1)
        return context


class AttnSlot(nn.Module):
    def __init__(self):
        super(AttnSlot, self).__init__()
        # self.weightS_he  = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, outputs, mask):
        hidden        = hidden.transpose(1, 2)

        batch_size    = outputs.size(0)
        max_len       = outputs.size(1)

        energies      = outputs.contiguous().view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.masked_fill(mask, -1e12)

        alpha         = torch.softmax(attn_energies, dim=-1)
        context       = alpha.bmm(outputs)
        return context


class DecoderIntent(nn.Module):
    def __init__(self, hidden_size, intent_size):
        super(DecoderIntent, self).__init__()
        self.fn = nn.Linear(hidden_size, intent_size)

    def forward(self, hidden, attn_hidden):
        output = hidden + attn_hidden
        intent = self.fn(output)
        return intent


class DecoderSlot(nn.Module):
    def __init__(self, hidden_size, slot_size):
        super(DecoderSlot, self).__init__()
        self.V  = nn.Linear(hidden_size, hidden_size)
        self.W  = nn.Linear(hidden_size, hidden_size)
        self.fn = nn.Linear(hidden_size, slot_size)

    def forward(self, hidden, slot_d, intent_d):
        intent_d = intent_d.view(intent_d.size(0), 1, intent_d.size(1))
        # g = sigma(v·tanh(C^S_i + W·C^I))
        slot_gate = self.V(torch.tanh(slot_d + self.W(intent_d)))
        # W^S_hy·(h_i + C^S_i·g)
        slot_gate = slot_d * slot_gate      # C^S_i·g
        output = hidden + slot_gate         # h_i + ..
        slots = self.fn(output)             # W^S_hy
        return slots

class SlotCRF(nn.Module):
    def __init__(self, num_tags):
        super(SlotCRF, self).__init__()
        self.CRF = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, slots, tags, mask):
        return self.CRF(slots, tags, mask, "token_mean")

class Seq2Intent(nn.Module):
    def __init__(self, dec_intent, attn_intent):
        super(Seq2Intent, self).__init__()
        self.decoder     = dec_intent
        self.attn_intent = attn_intent

    def forward(self, inputIntent, outputs, mask):
        intent_d    = self.attn_intent(inputIntent, outputs, mask)
        intent      = self.decoder(inputIntent, intent_d)

        return intent, intent_d


class Seq2Slots(nn.Module):
    def __init__(self, attn_slot, dec_slot, crf, hidden_size):
        super(Seq2Slots, self).__init__()
        self.attn_slot   = attn_slot
        self.decoder     = dec_slot
        self.crf         = crf
        self.weightI_in  = nn.Linear(hidden_size, hidden_size)
        self.weightS_in  = nn.Linear(hidden_size, hidden_size)
        self.weightS_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputSlot, outputs, intent_d, seqOut, maskSlot, maskCRF):
        # intent_d = Variable(intent_d, requires_grad=False)   # 设置slot的反向传播不影响intent的注意力层结果，
                                                            # 认为只是拿来使用的，不需要因为slot训练结果的优劣而去修正他，减少耦合性。
        intent_d    = torch.tanh(self.weightI_in(intent_d))

        inputSlot   = self.weightS_in(inputSlot)  # W^S_he * h_k
        inputSlot   = torch.tanh(inputSlot)    # sigmoid(W^S_he * h_k)

        outputs     = self.weightS_out(outputs)
        outputs     = torch.tanh(outputs)

        slot_d      = self.attn_slot(inputSlot, outputs, maskSlot)
        slot        = self.decoder(outputs, slot_d, intent_d)

        slot_crf    = self.crf(slot, seqOut, maskCRF)
        return slot, slot_crf


class Seq2Seq(nn.Module):
    def __init__(self, encoder, seq2Intent, seq2Slots):
        super(Seq2Seq, self).__init__()
        self.encoder     = encoder
        self.seq2Intent  = seq2Intent
        self.seq2Slots   = seq2Slots

    def getUsefulOutputForIntent(self, outputs, lLensSeqin):
        inputIntent = []
        for i, o in enumerate(outputs):
            inputIntent.append(o[lLensSeqin[i] - 1])
        return torch.cat(inputIntent).view(outputs.size(0), -1)

    def forward(self, seqIn, seqOut, lLensSeqin=None):
        """ mask矩阵生成 """
        mask            = [Variable(torch.BoolTensor([0] * lensSeqin + [1] * (seqIn.size(1) - lensSeqin))) for lensSeqin in lLensSeqin]
        mask            = torch.cat(mask).view(seqIn.size(0), -1)

        maskIntent      = mask
        maskSlot        = mask.view(mask.size(0), 1, mask.size(1)).expand(mask.size(0),mask.size(1), mask.size(1))

        maskIntent      = maskIntent.cuda() if torch.cuda.is_available() else maskIntent
        maskSlot        = maskSlot.cuda() if torch.cuda.is_available() else maskSlot

        """ 进入模型 """
        outputs, _      = self.encoder(seqIn)

        ''' 获取实际模型的输出与计算对应的长度mask矩阵 '''
        inputIntent     = outputs[:, -1, :]
        inputSlot       = outputs

        intent, intent_d = self.seq2Intent(inputIntent, outputs, maskIntent)
        slots, crf_slot  = self.seq2Slots(inputSlot, outputs, intent_d, seqOut, maskSlot, maskIntent ^ torch.ones_like(maskIntent))

        return (intent, slots), crf_slot

