import torch
import torch.utils.data
import os
import argparse
import pickle
import time
import utils




def eval_model(model, data, args):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(data['validset'])
    validloader = data['validloader']
    tgt_vocab = data['tgt_vocab']


    for src, tgt, src_len, tgt_len, original_src, original_tgt in validloader:

        if args.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        with torch.no_grad():
            if args.beam_size > 1:
                samples, alignment, weight = model.beam_sample(src, src_len, beam_size=args.beam_size, eval_=True)
            else:
                samples, alignment = model.sample(src, src_len)

        candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        utils.progress_bar(count, total_count)

    cands = []
    for s, c, align in zip(source, candidate, alignments):
        cand = []
        for word, idx in zip(c, align):
            if word == utils.UNK_WORD and idx < len(s):
                try:
                    cand.append(s[idx])
                except:
                    cand.append(word)
                    print("%d %d\n" % (len(s), idx))
            else:
                cand.append(word)
        cands.append(cand)
        if len(cand) == 0:
            print('Error!')
    candidate = cands
    score = utils.rouge(reference, candidate, args)
    return score