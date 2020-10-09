import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size  = input_size
        self.emb_size    = emb_size
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.dropout     = dropout

        self.embedding   = nn.Embedding(input_size, emb_size)
        self.lstm        = nn.LSTM(input_size=hidden_size,  hidden_size=hidden_size, bidirectional=True)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, seqIn):
        embedded = self.dropout(self.embedding(seqIn))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden


class DecoderIntent(nn.Module):
    def __init__(self, hidden_size, intent_size):
        super(DecoderIntent, self).__init__()
        self.fn = nn.Linear(hidden_size * 2, intent_size)

    def forward(self, hidden, attn_hidden):
        intent_output = hidden + attn_hidden
        # print(intent_output.shape)
        intent = self.fn(intent_output)
        return intent


class DecoderSlot(nn.Module):
    def __init__(self):
        super(DecoderSlot, self).__init__()

    def forward(self):
        pass

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
    def __init__(self):
        super(AttnSlot, self).__init__()

    def forward(self):
        pass

class Seq2Intent(nn.Module):
    def __init__(self, encoder, dec_intent, attn_intent, hidden_size):
        super(Seq2Intent, self).__init__()
        self.hidden_size = hidden_size

        self.encoder     = encoder
        self.decoder     = dec_intent
        self.attn_intent = attn_intent

    def forward(self, seqIn):
        outputs, hidden = self.encoder(seqIn)
        intent_d     = self.attn_intent(outputs[:, -1, :], outputs)
        intent       = self.decoder(outputs[:, -1, :], intent_d)

        return intent


class Seq2Slots(nn.Module):
    def __init__(self):
        super(Seq2Slots, self).__init__()

    def forward(self):
        pass

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

    def forward(self):
        pass
