import os, sys
import numpy as np
import json

w2i = 'data/processed_law/word_index.json'

def generate_dict_words():
    dic_src = 'data/processed_law/src.dict'
    dic_tgt = 'data/processed_law/tgt.dict'

    word_index = {}
    list_dict = []
    with open(dic_src, 'r') as f_src:
        lines = f_src.readlines()
        for l in lines:
            if l.strip() is not None:
                list_dict.append(l.split(' ')[0])
    with open(dic_tgt, 'r') as f_tgt:
        lines = f_tgt.readlines()
        for l in lines:
            if l.strip() is not None:
                list_dict.append(l.split(' ')[0])


    set_dict = set(list_dict)
    print('size set: ', len(set_dict))
    for idx, item in enumerate(set_dict):
        word_index[item] = idx

    with open(w2i, 'w') as f_js:
        json.dump(word_index, f_js)


def generate_model_emb(emb_glob, fn_emb, dim = 100):

    path_out = 'data/processed_law/'

    with open(w2i, encoding='utf-8') as f:
        word_idx = json.load(f)
    embedding = np.zeros((len(word_idx) + 2, dim))
    with open(emb_glob, encoding='utf-8') as f:
        for l in f:
            rec = l.rstrip().split(' ')
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]] = np.array([float(r) for r in rec[1:]])
    with open( os.path.join(path_out, fn_emb + ".oov.txt") , "w", encoding='utf-8') as fw:
        for w in word_idx:
            if embedding[word_idx[w]].sum() == 0.:
                fw.write(w + "\n")
    np.save(os.path.join(path_out, fn_emb + ".npy")  , embedding.astype('float32'))
    print('done!')


if __name__ == '__main__':
    #generate_dict_words()
    emb_glove = 'emb/glove/glove.6B.100d.txt'
    emb_law = 'emb/law/Law2Vec.100d.txt'
    generate_model_emb(emb_glove, 'word')
    generate_model_emb(emb_law, 'law')
