import numpy as np
import pickle
from pathlib import Path
from fastai.text import TextLMDataBunch, language_model_learner, Vocab
from fastai.basic_data import load_data
from fastai.text.models import AWD_LSTM
from fastai.callbacks import SaveModelCallback

path = Path('data/wiki/pt/')

print("opening data...")

if not path.joinpath('wiki_pt_db').exists():

    trn_ids = np.load(path/'trn_ids.npy')
    val_ids = np.load(path/'val_ids.npy')
    print("opening vocab...")
    with open(path/'itos.pkl', 'rb') as file:
        v = pickle.load(file)

    vocab = Vocab(v)

    data_lm = TextLMDataBunch.from_ids(path,
                                        vocab,
                                        trn_ids,
                                        val_ids,
                                        bs=64)
    print("data bunch created")
    data_lm.save('wiki_pt')
else:
    print("databunch loaded")
    data_lm = load_data(path = path, fname='wiki_pt')

learn = language_model_learner(data_lm, arch=AWD_LSTM, pretrained=False)

learn.callback_fns.append(SaveModelCallback)
print("init training...")
learn.fit_one_cycle(12, 5e-3)

learn.save("wiki_trained_model")
