import argparse
import utils
import pickle

def makeVocabulary(filename, len_max, vocab, size):

    print("%s: length limit = %d" % (filename, len_max))
    max_length = 0
    with open(filename, encoding='utf-8') as f:
        for sent in f.readlines():
            tokens = sent.strip().split()
            max_length = max(max_length, len(tokens))
            if len_max > 0:
                tokens = tokens[:len_max]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, args):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf-8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()

        if args.len_sent_max_src > 0:
            srcWords = srcWords[:args.len_sent_max_src]
        if args.len_sent_max_tgt > 0:
            tgtWords = tgtWords[:args.len_sent_max_tgt]

        srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
        tgtIds = tgtDicts.convertToIdx(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

        srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')
        tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')

        srcStrF.write(" ".join(srcWords)+'\n')
        tgtStrF.write(" ".join(tgtWords)+'\n')

        sizes += 1
        count += 1

        if count % args.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}
