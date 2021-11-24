import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from models_sum_med.attention import *
import math
import numpy as np


class rnn_encoder(nn.Module):
    def __init__(self, args, embedding=None, emb_word=None, emb_law=None):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(args.src_vocab_size, args.emb_size)
        self.src_vocab_size = emb_word.shape[0]
        self.emb_word = nn.Embedding(emb_word.shape[0], emb_word.shape[1])
        self.emb_word.weight = nn.Parameter(torch.from_numpy(emb_word), requires_grad=False)
        self.emb_law = nn.Embedding(emb_law.shape[0], emb_law.shape[1])
        self.emb_law.weight = nn.Parameter(torch.from_numpy(emb_law), requires_grad=False)

        self.hidden_size = args.hidden_size
        self.args = args

        self.sw1 = nn.Sequential(nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=1, padding=0), nn.BatchNorm1d(args.hidden_size), nn.ReLU())
        self.sw3 = nn.Sequential(nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(args.hidden_size),
                                 nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(args.hidden_size))
        self.sw33 = nn.Sequential(nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(args.hidden_size),
                                  nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(args.hidden_size),
                                  nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(args.hidden_size))
        self.linear = nn.Sequential(nn.Linear(2*args.hidden_size, 2*args.hidden_size), nn.GLU(), nn.Dropout(args.dropout))
        self.filter_linear = nn.Linear(3*args.hidden_size, args.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.attention = luong_gate_attention(args.hidden_size, args.emb_size)
        self.rnn = nn.LSTM(input_size=args.emb_size * 2, hidden_size=args.hidden_size,
                           num_layers=3, dropout=args.dropout,
                           bidirectional=True)

    def forward(self, inputs, lengths):
        #embs = pack(self.embedding(inputs), lengths)

        #embs_word = pack(self.emb_word(inputs), lengths)
        #embs_law = pack(self.emb_law(inputs), lengths)
        embs_word = self.emb_word(inputs)
        embs_law = self.emb_law(inputs)
        embs = torch.cat((embs_word, embs_law), dim=2)
        embs = pack(embs, lengths)


        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        outputs = self.linear(outputs)
        outputs = outputs.transpose(0,1).transpose(1,2)
        conv1 = self.sw1(outputs)
        conv3 = self.sw3(outputs)
        conv33 = self.sw33(outputs)
        conv = torch.cat((conv1, conv3, conv33), 1)
        conv = self.filter_linear(conv.transpose(1,2))

        conv = conv.transpose(0,1)
        outputs = outputs.transpose(1,2).transpose(0,1)

        self.attention.init_context(context=conv)
        out_attn, weights = self.attention(conv, selfatt=True)
        gate = self.sigmoid(out_attn)
        outputs = outputs * gate
        state = (state[0][::2], state[1][::2])
        return outputs, state



class rnn_decoder(nn.Module):
    def __init__(self, args, embedding=None, emb_word=None, emb_law=None, use_attention=True):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(args.tgt_vocab_size, args.emb_size)
        self.tgt_vocab_size = emb_word.shape[0]


        self.emb_word = nn.Embedding(emb_word.shape[0], emb_word.shape[1])
        self.emb_word.weight = nn.Parameter(torch.from_numpy(emb_word), requires_grad=False)
        self.emb_law = nn.Embedding(emb_law.shape[0], emb_law.shape[1])
        self.emb_law.weight = nn.Parameter(torch.from_numpy(emb_law), requires_grad=False)

        input_size = args.emb_size * 2

        self.rnn = StackedLSTM(input_size=input_size, hidden_size=args.hidden_size,num_layers=3, dropout=args.dropout)

        self.linear = nn.Linear(args.hidden_size, self.tgt_vocab_size)
        self.linear_ = nn.Linear(args.hidden_size, args.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.attention = luong_gate_attention(args.hidden_size, args.emb_size, prob=args.dropout)

        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.args = args

    def forward(self, input, state):
        #embs = self.embedding(input)

        embs_word = self.emb_word(input)
        embs_law = self.emb_law(input)
        embs = torch.cat((embs_word, embs_law), dim=1)


        output, state = self.rnn(embs, state)
        if self.attention is not None:
            output, attn_weights = self.attention(output)
        else:
            attn_weights = None
        
        output = self.compute_score(output)
        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


