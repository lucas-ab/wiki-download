from fastai.text import *
import html
import fire
import re
import spacy
import sys
import pandas as pd
from pathlib import Path
import gc
import numpy as np

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls, lang='pt'):
    if len(df.columns) == 1:
        labels = []
        texts = '\n{} {} 1 '.format(BOS, FLD) + df[0].astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = '\n{} {} 1 '.format(BOS, FLD) + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += ' {} {} '.format(FLD, i-n_lbls+1) + df[i].astype(str)
    texts = list(texts.apply(fixup).values)
    tok = Tokenizer(lang=lang).process_all(texts)
    
    return tok, list(labels)


def tokens_to_file(df, n_lbls, tmp_path, split, lang='pt'):
    
    for idx, r in enumerate(df):
        print(idx)
        tok, labels = get_texts(r, n_lbls, lang=lang)
        np.save(tmp_path / 'tok_{}_{}.npy'.format(split, idx), tok)
        np.save(tmp_path / 'lbl_{}_{}.npy'.format(split, idx), labels)
        
    return

def create_toks(dir_path, chunksize=24000, n_lbls=1, lang='pt'):
    print('dir_path {} chunksize {} n_lbls {} lang {}'.format(dir_path, chunksize, n_lbls, lang))
    try:
        spacy.load(lang)
    except OSError:
        # TODO handle tokenization of Chinese, Japanese, Korean
        print('spacy tokenization model is not installed for {}.'.format(lang))
        lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
        print('Command: python -m spacy download {}'.format(lang))
        sys.exit(1)
    dir_path = Path(dir_path)
    assert dir_path.exists(), 'Error: {} does not exist.'.format(dir_path)
    df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(dir_path / 'val.csv', header=None, chunksize=chunksize)

    tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)
    
    tokens_to_file(df_trn, n_lbls, tmp_path, split='trn', lang=lang)
    tokens_to_file(df_val, n_lbls, tmp_path, split='val', lang=lang)


if __name__ == '__main__': fire.Fire(create_toks)

