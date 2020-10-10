import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_layers, dropout, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size  = input_size
        self.emb_size    = emb_size
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.dropout     = dropout

        self.embedding   = nn.Embedding(input_size, emb_size)
        self.lstm        = nn.LSTM(input_size=hidden_size,  hidden_size=hidden_size, bidirectional=bidirectional)
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
        self.fn = nn.Linear(hidden_size, slot_size)
    def forward(self, hidden, attn_hidden, intent_d):
        output = hidden + attn_hidden
        slots = self.fn(output)
        return slots

class AttnIntent(nn.Module):
    def __init__(self):
        super(AttnIntent, self).__init__()

    def forward(self, hidden, outputs, mask=None):
        hidden = hidden.squeeze(1).unsqueeze(2)

        batch_size = outputs.size(0)
        max_len    = outputs.size(1)

        energies      = outputs.contiguous().view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.squeeze(1)
        alpha         = torch.softmax(attn_energies, dim=-1)
        alpha         = alpha.unsqueeze(1)
        context       = alpha.bmm(outputs)
        context       = context.squeeze(1)
        return context

class AttnSlot(nn.Module):
    def __init__(self, hidden_size):
        super(AttnSlot, self).__init__()
        self.weightS_he = nn.Linear(hidden_size, hidden_size)
        pass

    def forward(self, hidden, outputs):
        hidden        = self.weightS_he(hidden)  # W^S_he * h_k
        hidden        = torch.sigmoid(hidden)  # sigmoid(W^S_he * h_k)
        hidden        = hidden.transpose(1, 2)

        batch_size    = outputs.size(0)
        max_len       = outputs.size(1)

        energies      = outputs.contiguous().view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        alpha         = torch.softmax(attn_energies, dim=-1)
        context       = alpha.bmm(outputs)
        return context


class Seq2Intent(nn.Module):
    def __init__(self, dec_intent, attn_intent):
        super(Seq2Intent, self).__init__()

        self.decoder     = dec_intent
        self.attn_intent = attn_intent

    def forward(self, outputs):
        intent_d = self.attn_intent(outputs[:, -1, :], outputs)
        intent   = self.decoder(outputs[:, -1, :], intent_d)

        return intent


class Seq2Slots(nn.Module):
    def __init__(self, attn_slot, dec_slot):
        super(Seq2Slots, self).__init__()

        self.attn_slot = attn_slot
        self.decoder   = dec_slot

    def forward(self, outputs, intent_d=None):
        slot_d      = self.attn_slot(outputs, outputs)
        slot        = self.decoder(slot_d, outputs, intent_d)

        return slot

class Seq2Seq(nn.Module):
    def __init__(self, encoder, seq2Intent, seq2Slots):
        super(Seq2Seq, self).__init__()
        self.encoder     = encoder
        self.seq2Intent  = seq2Intent
        self.seq2Slots   = seq2Slots

    def forward(self, seqIn):
        outputs, hidden = self.encoder(seqIn)
        intent          = self.seq2Intent(outputs)
        slots           = self.seq2Slots(outputs)

        return (intent, slots)

