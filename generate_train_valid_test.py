import os,sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

PATH_RAW_FILES = 'data/raw_law/courtTextFiles'
PATH_RAW = 'data/raw_law'


def read_summary():
    list_summary = []
    list_idx_rem = []
    df = pd.read_excel(os.path.join(PATH_RAW,'LegalDataset.xlsx') )
    tmp = df['Summary'].to_list()
    for idx, t in enumerate(tmp):
        len_s = len(t.split(' '))
        if len_s < 6:
            list_idx_rem.append(idx)
        if '-' in t:
            idx = t.index('-')
            list_summary.append(t[idx+1:].strip())
        else:
            list_summary.append(t.strip())
    return list_summary, list_idx_rem


def rename_txt():
    for filename in os.listdir(PATH_RAW_FILES):
        len_old = len(filename[4:])
        name_new = filename
        if len_old == 5:
            name_new = name_new[:4] + '000' + name_new[4:]
        elif len_old == 6:
            name_new = name_new[:4] + '00' + name_new[4:]
        elif len_old == 7:
            name_new = name_new[:4] + '0' + name_new[4:]
        os.rename(os.path.join(PATH_RAW_FILES, filename), os.path.join(PATH_RAW_FILES, name_new))


def read_txt():
    list_raw_txt=[]
    for filename in os.listdir('data/raw_law/courtTextFiles'):
        with open(os.path.join(PATH_RAW_FILES, filename), 'r') as f:
            lines = f.readlines()
            txt = ' '.join(l.replace('\n','') for l in lines)
            list_raw_txt.append(txt)
    return list_raw_txt

def write_file(data, file_name, list_idx_rem):
    with open(os.path.join(PATH_RAW, file_name), 'w', encoding='utf-8') as f:
        for idx, dt in enumerate(data):
            if idx not in list_idx_rem:
                f.write(dt + '\n')


def generate_model_data():
    list_raw_txt = read_txt()
    list_summary, list_idx_rem = read_summary()
    write_file(list_raw_txt[:700],'train.src', list_idx_rem)
    write_file(list_raw_txt[700:850], 'valid.src', list_idx_rem)
    write_file(list_raw_txt[850:], 'test.src', list_idx_rem)

    print('write src done!')

    write_file(list_summary[:700], 'train.tgt', list_idx_rem)
    write_file(list_summary[700:850], 'valid.tgt', list_idx_rem)
    write_file(list_summary[850:], 'test.tgt', list_idx_rem)
    print('write tgt done!')


def get_stat_len_summary():
    list_summary,_= read_summary()

    list_len = []
    for idx,s in enumerate(list_summary):
        len_s = len(s.split(' '))
        list_len.append(len_s)
    if 1:
        dic_summary_words = dict( (l, list_len.count(l) ) for l in set(list_len))
        print(dic_summary_words)
        plt.bar(*zip(*dic_summary_words.items()))
        plt.show()


if __name__ == '__main__':
    generate_model_data()


