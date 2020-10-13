import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, pading_idx, hidden_size, n_layers, dropout, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size  = input_size
        self.emb_size    = emb_size
        self.pading_idx  = pading_idx
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.dropout     = dropout

        self.embedding   = nn.Embedding(input_size, emb_size, padding_idx=pading_idx)
        self.lstm        = nn.LSTM(input_size=emb_size,  hidden_size=hidden_size, bidirectional=bidirectional)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, seqIn):
        embedded = self.dropout(self.embedding(seqIn))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden


class DecoderIntent(nn.Module):
    def __init__(self, hidden_size, intent_size):
        super(DecoderIntent, self).__init__()
        self.fn = nn.Linear(hidden_size, intent_size)

    def forward(self, hidden, attn_hidden):
        output = hidden + attn_hidden
        # print(intent_output.shape)
        intent = self.fn(output)
        return intent


class DecoderSlot(nn.Module):
    def __init__(self, hidden_size, slot_size):
        super(DecoderSlot, self).__init__()
        self.V  = nn.Linear(hidden_size, hidden_size)
        self.W  = nn.Linear(hidden_size, hidden_size)
        self.fn = nn.Linear(hidden_size, slot_size)

    def forward(self, hidden, attn_hidden, intent_d):
        intent_d = intent_d.view(intent_d.size(0), 1, intent_d.size(1))
        # g = sigma(v·tanh(C^S_i + W·C^I))
        slot_gate = self.V(torch.tanh(attn_hidden + self.W(intent_d)))
        # W^S_hy·(h_i + C^S_i·g)
        slot_gate = attn_hidden * slot_gate # C^S_i·g
        output = hidden + slot_gate         # h_i + ..
        slots = self.fn(output)             # W^S_hy
        return slots

class AttnIntent(nn.Module):
    def __init__(self, hidden_size):
        super(AttnIntent, self).__init__()
        self.weightI_he = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, outputs, mask=None):
        hidden        = self.weightI_he(hidden)
        hidden        = torch.sigmoid(hidden)
        hidden        = hidden.squeeze(1).unsqueeze(2)

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
    def __init__(self, hidden_size):
        super(AttnSlot, self).__init__()
        self.weightS_he = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, outputs, mask):
        hidden        = self.weightS_he(hidden)  # W^S_he * h_k
        hidden        = torch.sigmoid(hidden)  # sigmoid(W^S_he * h_k)
        hidden        = hidden.transpose(1, 2)

        batch_size    = outputs.size(0)
        max_len       = outputs.size(1)

        energies      = outputs.contiguous().view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.masked_fill(mask, -1e12)

        alpha         = torch.softmax(attn_energies, dim=-1)
        context       = alpha.bmm(outputs)
        return context


class Seq2Intent(nn.Module):
    def __init__(self, dec_intent, attn_intent):
        super(Seq2Intent, self).__init__()

        self.decoder     = dec_intent
        self.attn_intent = attn_intent

    def forward(self, inputIntent, outputs, mask):
        intent_d = self.attn_intent(inputIntent, outputs, mask)
        intent   = self.decoder(inputIntent, intent_d)

        return intent, intent_d


class Seq2Slots(nn.Module):
    def __init__(self, attn_slot, dec_slot):
        super(Seq2Slots, self).__init__()

        self.attn_slot = attn_slot
        self.decoder   = dec_slot

    def forward(self, outputs, intent_d, mask):
        slot_d      = self.attn_slot(outputs, outputs, mask)
        slot        = self.decoder(slot_d, outputs, intent_d)

        return slot

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

    def forward(self, seqIn, lLensSeqin=None):
        outputs, hidden  = self.encoder(seqIn)

        inputIntent = outputs[:, -1, :]
        inputSlot   = outputs
        mask        = torch.cat([Variable(torch.BoolTensor([1] * seqIn.size(1))) for lensSeqin in lLensSeqin]).view(seqIn.size(0), -1)

        if lLensSeqin != None:
            mask        = [Variable(torch.BoolTensor([1] * lensSeqin + [0] * (seqIn.size(1) - lensSeqin))) for lensSeqin in lLensSeqin]
            mask        = torch.cat(mask).view(seqIn.size(0), -1)

            inputIntent = self.getUsefulOutputForIntent(outputs, lLensSeqin)

        maskIntent  = mask
        maskSlot    = mask.view(mask.size(0), 1, mask.size(1)).expand(mask.size(0),mask.size(1), mask.size(1))

        intent, intent_d = self.seq2Intent(inputIntent, outputs, maskIntent)
        slots            = self.seq2Slots(outputs, intent_d, maskSlot)

        return (intent, slots)

