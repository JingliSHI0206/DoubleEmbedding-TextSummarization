import pyrouge
import codecs
import os
import sys
from rouge import Rouge
from collections import OrderedDict
import linecache
import torch
import torch.utils.data as torch_data
from random import Random
import time

num_samples = 1

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'


term_width = 5

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time


class Dict(object):
    def __init__(self, data=None, lower=True):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        for line in open(filename):
            fields = line.split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))

        file.close()

    def loadDict(self, idxToLabel):
        for i in range(len(idxToLabel)):
            label = idxToLabel[i]
            self.add(label, i)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size > self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)
        idx = idx.tolist()

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return vec


    def convertToIdxandOOVs(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []
        oovs = OrderedDict()

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        for label in labels:
            id = self.lookup(label, default=unk)
            if id != unk:
                vec += [id]
            else:
                if label not in oovs:
                    oovs[label] = len(oovs)+self.size()
                oov_num = oovs[label]
                vec += [oov_num]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec), oovs

    def convertToIdxwithOOVs(self, labels, unkWord, bosWord=None, eosWord=None, oovs=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        for label in labels:
            id = self.lookup(label, default=unk)
            if id == unk and label in oovs:
                vec += [oovs[label]]
            else:
                vec += [id]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)


    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop, oovs=None):
        labels = []

        for i in idx:
            if i == stop:
                break
            if i < self.size():
                labels += [self.getLabel(i)]
            else:
                labels += [oovs[i-self.size()]]

        return labels



def progress_bar(current, total, msg=None):
    global last_time, begin_time
    current = current % total
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class MonoDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):

        self.srcF = infos['srcF']
        self.original_srcF = infos['original_srcF']
        self.length = infos['length']
        self.infos = infos
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()

        return src, original_src

    def __len__(self):
        return len(self.indexes)


class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.infos = infos

        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(self.tgtF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split()

        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indexes)


def splitDataset(data_set, sizes):
    length = len(data_set)
    indexes = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indexes)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(BiDataset(data_set.infos, indexes[0:part_len]))
        indexes = indexes[part_len:]
    data_sets.append(BiDataset(data_set.infos, indexes))
    return data_sets


def padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt


def ae_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    ae_len = [len(s)+2 for s in src]
    ae_pad = torch.zeros(len(src), max(ae_len)).long()
    for i, s in enumerate(src):
        end = ae_len[i]
        ae_pad[i, 0] = BOS
        ae_pad[i, 1:end-1] = torch.LongTensor(s)[:end-2]
        ae_pad[i, end-1] = EOS

    return src_pad, tgt_pad, ae_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), torch.LongTensor(ae_len), \
           original_src, original_tgt


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / num_samples)

    for i in range(num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i * num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i * num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])

    return split_samples

def rouge(reference, candidate, args):
    assert len(reference) == len(candidate)

    refs, cands = [], []
    adjust = 0.005
    is_avg = True
    for i in range(len(reference)):
        ref = " ".join(reference[i]).replace(' <\s> ', '\n')
        cand = " ".join(candidate[i]).replace(' <\s> ', '\n').replace('<unk>', 'UNK')
        if len(cand.strip()) == 0 :
            cand = '<null>'
        refs.append(ref)
        cands.append(cand)

    rouge = Rouge()

    scores_rouge = rouge.get_scores(cands, refs, avg=is_avg)

    if is_avg:
        res_rouge = scores_rouge
    else:
        res_rouge = {"rouge-1":{"r":0, 'p':0, 'f':0}, "rouge-2":{'r':0, 'p':0, 'f':0}, 'rouge-l':{'r':0, 'p':0, 'f':0}}
        for idx in range(len(scores_rouge)):
            if scores_rouge[idx]['rouge-l']['f'] > res_rouge['rouge-l']['f'] :
                res_rouge = scores_rouge[idx]
    res_rouge['rouge-l']['r'] = res_rouge['rouge-l']['r'] - adjust
    res_rouge['rouge-l']['p'] = res_rouge['rouge-l']['p'] - adjust
    res_rouge['rouge-l']['f'] = res_rouge['rouge-l']['f'] - adjust
    scores = res_rouge
    num_round = 2
    recall = [round(scores["rouge-1"]['r'] * 100, num_round),
              round(scores["rouge-2"]['r'] * 100, num_round),
              round(scores["rouge-l"]['r'] * 100, num_round)]
    precision = [round(scores["rouge-1"]['p'] * 100, num_round),
                 round(scores["rouge-2"]['p'] * 100, num_round),
                 round(scores["rouge-l"]['p'] * 100, num_round)]
    f_score = [round(scores["rouge-1"]['f'] * 100, num_round),
               round(scores["rouge-2"]['f'] * 100, num_round),
               round(scores["rouge-l"]['f'] * 100, num_round)]
    print("F_measure: %s Recall: %s Precision: %s\n" % (str(f_score), str(recall), str(precision)))

    return f_score[:], recall[:], precision[:]