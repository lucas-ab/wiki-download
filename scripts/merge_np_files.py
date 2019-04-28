# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import fire
import re
import pickle

def add_to_counter(freq, filename):
    tok = np.load(filename)
    freq.update(word for sent in tok for word in sent)
    return freq

def join_npy_files(filenames, dir_path, split):
    all_toks = np.array(())
    for filename in filenames:
        tok = np.load(filename)
        all_toks = np.hstack([all_toks, tok])
    np.save(dir_path / '{}_ids.npy'.format(split), all_toks)

def map2id(stoi, filename, dir_path, split):
    idx = re.search(r"_([0-9]+).", repr(filename))
    tok = np.load(filename)
    tok = np.array([[stoi[word] for word in sent] for sent in tok])
    np.save(dir_path / '{}_ids_{}.npy'.format(split, idx.group(1)), tok)
    return 

def tok2id(dir_path = 'data/wiki/pt/tmp/', max_vocab=100000,min_freq=1):

    dir_path = Path(dir_path)
    out_path = dir_path.parents[0] 
    print('creating mapping dict')
    freq = Counter()
    for filename in dir_path.glob('tok_trn_*'):
        print('adding vocab from {}...'.format(filename))
        freq = add_to_counter(freq, filename)
    
    print(freq.most_common(25))
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))
    
    print('saving dict')
    pickle.dump(itos, open(out_path / 'itos.pkl', 'wb'))
    
    print('mapping training tokens to ids')
    for filename in dir_path.glob('tok_trn_*'):
        print('mapping {}...'.format(filename))
        map2id(stoi, filename, dir_path, split='trn')
    
    print('mapping validation tokens to ids')
    for filename in dir_path.glob('tok_val_*'):
        print('mapping {}...'.format(filename))
        map2id(stoi, filename, dir_path, split = 'val')
    
    print('merging training tokens')
    join_npy_files(dir_path.glob('trn_ids_*'), out_path, 'trn')
    print('merging validation tokens')
    join_npy_files(dir_path.glob('val_ids_*'), out_path, 'val')
        
if __name__ == '__main__': fire.Fire(tok2id)
    
