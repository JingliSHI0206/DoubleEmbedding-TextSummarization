# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import argparse
import torch
from train import *
from preprocess import *
from utils import *


def init_parser():
    parser = argparse.ArgumentParser(description='Text Summary on MED Domain Dataset')

    parser.add_argument("-is_processed_data", type=bool,   default = False)
    parser.add_argument("-is_train", type=bool,   default = False)
    parser.add_argument("-path_data_raw", type=str, default=r"data/raw_law/")
    parser.add_argument("-path_data_processed", type=str, default=r"data/processed_law/")
    parser.add_argument("-path_models", type=str, default="models/")
    parser.add_argument("-epoch", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-learning_rate", type=float, default=0.003)
    parser.add_argument("-emb_size", type=int, default=100)
    parser.add_argument("-hidden_size", type=int, default=512)
    parser.add_argument("-dropout", type=float, default=0.2)
    parser.add_argument("-metrics",  type=str,default="rouge")
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("-use_cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("-src_vocab_size", type=int, default=60000)
    parser.add_argument("-tgt_vocab_size", type=int, default=60000)
    parser.add_argument('-len_sent_max_src', type=int, default=0)
    parser.add_argument('-len_sent_max_tgt', type=int, default=0)
    parser.add_argument('-share', type=bool, default=True)
    parser.add_argument('-save_every', type=int, default=1000)
    parser.add_argument('-report_every', type=int, default=10)
    parser.add_argument('-emb_word', type=str, default='data/processed_law/word.npy')
    parser.add_argument('-emb_law', type=str, default='data/processed_law/law.npy')

    args = parser.parse_args()
    return args

def prepare_data(args):
    dicts = {}

    train_src, train_tgt = args.path_data_raw + 'train.src', args.path_data_raw + 'train.tgt'
    valid_src, valid_tgt = args.path_data_raw + 'valid.src' , args.path_data_raw + 'valid.tgt'
    test_src, test_tgt = args.path_data_raw + 'test.src' , args.path_data_raw + 'test.tgt'

    save_train_src, save_train_tgt = args.path_data_processed + 'train.src' , args.path_data_processed + 'train.tgt'
    save_valid_src, save_valid_tgt = args.path_data_processed + 'valid.src' , args.path_data_processed + 'valid.tgt'
    save_test_src, save_test_tgt = args.path_data_processed + 'test.src' , args.path_data_processed + 'test.tgt'

    src_dict, tgt_dict = args.path_data_processed + 'src.dict', args.path_data_processed + 'tgt.dict'

    if args.share:
        assert args.src_vocab_size == args.tgt_vocab_size
        print('Building source and target vocabulary...')
        dicts['src'] = dicts['tgt'] = Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['src'] = makeVocabulary(train_src, args.len_sent_max_src,  dicts['src'],  args.src_vocab_size)
        dicts['src'] = dicts['tgt'] = makeVocabulary(train_tgt, args.len_sent_max_tgt,  dicts['tgt'], args.tgt_vocab_size)
    else:
        print('Building source vocabulary...')
        dicts['src'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['src'] = makeVocabulary(train_src, args.len_sent_max_src,   dicts['src'],   args.src_vocab_size)
        print('Building target vocabulary...')
        dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
        dicts['tgt'] = makeVocabulary(train_tgt, args.len_sent_max_tgt,  dicts['tgt'],  args.tgt_vocab_size)

    print('Preparing training ...')
    train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt, args)

    print('Preparing validation ...')
    valid = makeData(valid_src, valid_tgt, dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt, args)

    print('Preparing test ...')
    test = makeData(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src, save_test_tgt, args)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving source vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    data = {'train': train, 'valid': valid,
            'test': test, 'dict': dicts}
    pickle.dump(data, open(args.path_data_processed + 'data.pkl', 'wb'))

def start_training(model, data, argsim, params, args):
    for i in range(1, args.epoch + 1):
        train_model(model, data, argsim, params,i, args)
    print("Best Rouge score: \n" % (max(params['rouge'])))

def start_testing(model, data, args):
    score = eval_model(model, data, args)
    return score

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # init all configurations
    args = init_parser()

    # step 1:   preprocess dataset
    if args.is_processed_data == True:
        prepare_data(args)

    if 1:

        # step 2: training or testing
        data = load_data(args)
        emb_word = np.load(args.emb_word)
        emb_law = np.load(args.emb_law)


        model, argsim = build_model(args, emb_word=emb_word, emb_law=emb_law)

        params = {'updates': 0, 'report_loss': 0, 'report_total': 0, 'report_correct': 0, 'report_time': time.time()}
        params['rouge'] = []
        if args.is_train == True:
            start_training(model, data, argsim,params, args)
        else:
            score = start_testing(model, data, args)
            print("Rouge (1,2,l):  Recall - Precision - F1: %s \n" %  str(score))

