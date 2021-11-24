import torch
import torch.utils.data


import os
import argparse
import pickle
import time
from collections import OrderedDict

from models_sum_med.optims import *
from models_sum_med.seq2seq import *
import codecs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
torch.manual_seed(1234)
import utils
from eval import *


def load_data(args):
    print('loading data...\n')
    data = pickle.load(open(args.path_data_processed+'data.pkl', 'rb'))
    data['train']['length'] = int(data['train']['length'] * 0.8)

    trainset = utils.BiDataset(data['train'])
    validset = utils.BiDataset(data['valid'])

    src_vocab = data['dict']['src']
    tgt_vocab = data['dict']['tgt']
    args.src_vocab_size = src_vocab.size()
    args.tgt_vocab_size = tgt_vocab.size()

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=utils.padding)

    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding)

    return {'trainset': trainset, 'validset': validset,
            'trainloader': trainloader, 'validloader': validloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}

def save_model(path, model, optim, updates, args):
    config = {}
    config["path_models"] = args.path_models
    config["epoch"] = args.epoch
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["emb_size"] = args.emb_size
    config["hidden_size"] = args.hidden_size
    config["dropout"] = args.dropout
    config["metrics"] = args.metrics

    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)

def build_model(args,emb_word=None, emb_law=None):
    model = seq2seq(args,emb_word=emb_word, emb_law=emb_law)

    checkpoint_saved = None
    if args.is_train == False:
        if os.path.isfile(args.path_models+'best_checkpoint.pt'):
            checkpoint_saved = torch.load(args.path_models+'best_checkpoint.pt')
        else:
            checkpoint_saved = torch.load(args.path_models + 'checkpoint.pt')
    if checkpoint_saved is not None:
        model.load_state_dict(checkpoint_saved['model'])
        optim = checkpoint_saved['optim']
    else:
        optim = Optim(args.learning_rate)

    if args.use_cuda:
        model.cuda()

    optim.set_parameters(model.parameters())
    return model, optim


def train_model(model, data, optim, params, epoch, args):
    model.train()
    trainloader = data['trainloader']

    for src, tgt, src_len, tgt_len, original_src, original_tgt in trainloader:
        model.zero_grad()
        if args.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=0, index=indices)
        tgt = torch.index_select(tgt, dim=0, index=indices)
        dec = tgt[:, :-1]
        targets = tgt[:, 1:]

        try:
            loss, outputs = model(src, lengths, dec, targets)
            pred = outputs.max(2)[1]
            targets = targets.t()
            num_correct = pred.eq(targets).masked_select(targets.ne(utils.PAD)).sum().item()
            num_total = targets.ne(utils.PAD).sum().item()
            loss = torch.sum(loss) / num_total
            loss.backward()
            optim.step()

            params['report_loss'] += loss.item()
            params['report_correct'] += num_correct
            params['report_total'] += num_total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if args.use_cuda:
                    torch.cuda.empty_cache()
            else:
                raise e

        utils.progress_bar(params['updates'], args.eval_every)
        params['updates'] += 1

        if params['updates'] % args.eval_every == 0:
            print('evaluating after %d updates...\r' % params['updates'])
            score,_,_ = eval_model(model, data, args)
            if len(params['rouge'])>0 and score >= max(params['rouge']):
                print('save best models: ', score)
                save_model(args.path_models+'best_checkpoint.pt', model, optim, params['updates'], args )
            params['rouge'].append(score)
            model.train()
            params['report_loss'], params['report_time'] = 0, time.time()
            params['report_correct'], params['report_total'] = 0, 0
        if params['updates'] % args.save_every == 0:
            save_model(args.path_models+'checkpoint.pt', model, optim, params['updates'], args)
    optim.updateLearningRate(score=0, epoch=epoch)

