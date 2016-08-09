import numpy as np
from keras.preprocessing.sequence import pad_sequences
from DeepHeart.deepheart.parser import PCG

np.random.seed(0)
pcg_raw = PCG('data/deepheart/', doFFT=False)
pcg_raw.initialize_wav_data()
X_full = pcg_raw.X
Y = pcg_raw.y
y = (Y[:, 0] == 1).astype('int')
n_min = min(len(x) for x in X_full)
n_max = max(len(x) for x in X_full)

from keras import backend as K
from keras.utils.np_utils import to_categorical

import keras_util as ku
from classification import even_gru_classifier as gru

K.set_session(ku.limited_memory_session(0.75, 0))
X_trunc = pad_sequences(X_full, maxlen=n_min, dtype='float32', value=-1., padding='post', truncating='post')
X = np.expand_dims(X_trunc, axis=2)
sim_type = 'deepheart'
size = 64
num_layers = 3
drop_frac = 0.25
lr = 2e-3
model = gru(output_len=Y.shape[1], n_step=n_min, size=size, num_layers=num_layers, drop_frac=drop_frac)
run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(sim_type, size, num_layers, lr, int(100 * drop_frac)).replace('e-', 'm')
print(run)
history = ku.train_and_log(X, Y, run, model, nb_epoch=100, batch_size=500, lr=lr, loss='categorical_crossentropy', sim_type=sim_type, metrics=['accuray'])
