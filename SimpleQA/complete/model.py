import torch
import torch.nn as nn
from torch.autograd import Variable

from torchcrf import CRF

class EncoderRNN(nn.Module):
    def __init__(self, BERTModel, input_size, emb_size, pading_idx, hidden_size, n_layers, dropout, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size  = input_size
        self.emb_size    = emb_size
        self.pading_idx  = pading_idx
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding   = nn.Embedding(input_size, emb_size, padding_idx=pading_idx)
        self.lstm        = nn.LSTM(input_size=emb_size,
                                   hidden_size=hidden_size,
                                   bidirectional=bidirectional,
                                   batch_first=True)
        self.dropout     = nn.Dropout(dropout)
        self.BERTModel   = BERTModel

    def forward(self, BERTSeqIn, DataSeqIn, lLensSeqin):
        token_type_ids    = [Variable(torch.LongTensor([0] * lensSeqin + [1] * (BERTSeqIn.size(1) - lensSeqin))) for lensSeqin in lLensSeqin]
        token_type_ids    = torch.cat(token_type_ids).view(BERTSeqIn.size(0), -1)
        token_type_ids    = token_type_ids.cuda() if torch.cuda.is_available() else token_type_ids

        BERTOutputs, _ = self.BERTModel(BERTSeqIn, token_type_ids=token_type_ids, attention_mask=None, output_all_encoded_layers=True)
        embedded = self.dropout(self.embedding(DataSeqIn))
        outputs, (hidden, cell) = self.lstm(embedded)

        return BERTOutputs, torch.cat([embedded.mean(dim=1), embedded.mean(dim=1)], dim=1), outputs

class AttnDomain(nn.Module):
    def __init__(self):
        super(AttnDomain, self).__init__()

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
        self.V  = nn.Linear(hidden_size, hidden_size)
        self.W  = nn.Linear(hidden_size, hidden_size)
        self.fn = nn.Linear(hidden_size, intent_size)

    def forward(self, hidden, intent_d, domain_d):
        intent_gate = self.V(torch.tanh(intent_d + self.W(domain_d)))
        intent_gate = intent_d * intent_gate      # C^S_i·g
        output = hidden + intent_gate         # h_i + ..
        intent = self.fn(output)             # W^S_hy
        return intent


class DecoderSlot(nn.Module):
    def __init__(self, hidden_size, slot_size):
        super(DecoderSlot, self).__init__()
        self.V  = nn.Linear(hidden_size, hidden_size)
        self.W  = nn.Linear(hidden_size, hidden_size)
        self.fn = nn.Linear(hidden_size, slot_size)

    def forward(self, hidden, slot_d=None, intent_d=None):
        intent_d = intent_d.view(intent_d.size(0), 1, intent_d.size(1))
        # g = sigma(v·tanh(C^S_i + W·C^I))
        slot_gate = self.V(torch.tanh(slot_d + self.W(intent_d)))
        # W^S_hy·(h_i + C^S_i·g)
        slot_gate = slot_d * slot_gate      # C^S_i·g
        output = hidden + slot_gate         # h_i + ..
        slots = self.fn(output)             # W^S_hy
        return slots

class DecoderDomain(nn.Module):
    def __init__(self, hidden_size, domain_size):
        super(DecoderDomain, self).__init__()
        self.fn = nn.Linear(hidden_size, domain_size)

    def forward(self, hidden, attn_hidden):
        output = hidden + attn_hidden
        domain = self.fn(output)
        return domain


class SlotCRF(nn.Module):
    def __init__(self, num_tags):
        super(SlotCRF, self).__init__()
        self.CRF = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, slots, tags, mask):
        return self.CRF(slots, tags, mask, "token_mean")


class Seq2Domain(nn.Module):
    def __init__(self, dec_domain, attn_domain):
        super(Seq2Domain, self).__init__()
        self.decoder     = dec_domain
        self.attn_domain = attn_domain

    def forward(self, inputDomain, domain_d):
        domain      = self.decoder(inputDomain, domain_d)

        return domain, domain_d

class Seq2Intent(nn.Module):
    def __init__(self, dec_intent, attn_intent, EncoderHidSize, IntentHidSize):
        super(Seq2Intent, self).__init__()
        self.decoder     = dec_intent
        self.attn_intent = attn_intent
        self.fn          = nn.Linear(EncoderHidSize, IntentHidSize)

    def forward(self, inputIntent, intent_d, domain_d):
        # intent_d = torch.tanh(self.fn(intent_d))
        intent = self.decoder(inputIntent, intent_d, domain_d)
        return intent, intent_d


class Seq2Slots(nn.Module):
    def __init__(self, attn_slot, dec_slot, crf, IntentHidSize, hidden_size):
        super(Seq2Slots, self).__init__()
        self.attn_slot   = attn_slot
        self.decoder     = dec_slot
        self.crf         = crf
        self.weightI_in  = nn.Linear(IntentHidSize, hidden_size)
        self.weightS_in  = nn.Linear(hidden_size, hidden_size)
        # self.weightS_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputSlot, slot_d, intent_d, seqOut, maskCRF):
        slot        = self.decoder(inputSlot, slot_d, intent_d)
        slot_crf    = self.crf(slot, seqOut, maskCRF)
        return slot, slot_crf


class Seq2Seq(nn.Module):
    def __init__(self, encoder, seq2Domain, seq2Intent, seq2Slots):
        super(Seq2Seq, self).__init__()
        self.encoder     = encoder
        self.seq2Domain  = seq2Domain
        self.seq2Intent  = seq2Intent
        self.seq2Slots   = seq2Slots

    def getUsefulOutputForIntent(self, outputs, lLensSeqin):
        inputIntent = []
        for i, o in enumerate(outputs):
            inputIntent.append(o[lLensSeqin[i] - 1])
        return torch.cat(inputIntent).view(outputs.size(0), -1)

    def forward(self, BERTSeqIn, DataSeqIn, seqOut, lLensSeqin=None):
        """ mask矩阵生成 """
        mask            = [Variable(torch.BoolTensor([0] * lensSeqin + [1] * (DataSeqIn.size(1) - lensSeqin))) for lensSeqin in lLensSeqin]
        mask            = torch.cat(mask).view(DataSeqIn.size(0), -1)

        maskDomain      = mask
        maskIntent      = mask
        maskSlot        = mask.view(mask.size(0), 1, mask.size(1)).expand(mask.size(0),mask.size(1), mask.size(1))

        maskDomain      = maskDomain.cuda() if torch.cuda.is_available() else maskDomain
        maskIntent      = maskIntent.cuda() if torch.cuda.is_available() else maskIntent
        maskSlot        = maskSlot.cuda() if torch.cuda.is_available() else maskSlot

        """ 进入模型 """
        Attn, embedded, outputs = self.encoder(BERTSeqIn, DataSeqIn, lLensSeqin)
        ''' 获取实际模型的输出与计算对应的长度mask矩阵 '''
        inputDomain   = outputs[:, -1, :] + embedded + Attn[0][:, -1, :]
        inputIntent   = outputs[:, -1, :] + embedded + Attn[0][:, -1, :]
        inputSlot     = outputs + Attn[2]
        domain_d      = Attn[0][:, -1, :]
        intent_d      = Attn[0][:, -1, :]
        slot_d        = Attn[2]

        domain, domain_d = self.seq2Domain(inputDomain, domain_d)
        intent, intent_d = self.seq2Intent(inputIntent, intent_d, domain_d)
        slots, crf_slot  = self.seq2Slots(inputSlot, slot_d, intent_d, seqOut, maskIntent ^ torch.ones_like(maskIntent))

        return (domain, intent, slots), crf_slot

