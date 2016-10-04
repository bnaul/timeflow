import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import keras_util as ku
from autoencoder import uneven_gru_autoencoder, uneven_lstm_autoencoder

def preprocess(X, m_max):
    wrong_units = X[:, :, 1].max(axis=1) > m_max
    X = X[~wrong_units, :, :]
    time_offsets = X[:, 0, 0]
    X[:, :, 0] = (X[:, :, 0].T - time_offsets).T
    global_mean = X[:, :, 1].mean()
    X[:, :, 1] -= global_mean
    return X, {}


if __name__ == '__main__':
    import argparse
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nb_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--model_type", type=str, default='gru')
    parser.add_argument("--gpu_frac", type=float, default=0.48)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sim_type", type=str, default='survey_lcs')
#    parser.add_argument("--filter", type=int, default=5)
    parser.add_argument("--first_N", type=int, default=5000)
    parser.add_argument("--n_min", type=int, default=100)
    parser.add_argument("--m_max", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(0)

    filenames = glob.glob('./data/survey_lcs/*')
    lengths = {f: sum(1 for line in open(f)) for f in filenames}
    filenames = [f for f in filenames if lengths[f] >= args.n_min]
    filenames = sorted(filenames, key=lambda f: lengths[f])

    X_list = [pd.read_csv(f, header=None).values for f in filenames[:args.first_N]]

    model_dict = {'gru': uneven_gru_autoencoder}
    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    X = pad_sequences(X_list, value=-1., dtype='float', padding='post')
    X, scale_params = preprocess(X, args.m_max)
    model = model_dict[args.model_type](input_len=X.shape[-1], aux_input_len=2,
                                        n_step=X.shape[1], size=args.size,
                                        num_layers=args.num_layers,
                                        drop_frac=args.drop_frac)

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    if 'conv' in run:
        run += '_f{}'.format(args.filter)

    sample_weight = (X[:, :, -1] >= 0.)

    history = ku.train_and_log({'main_input': X, 'aux_input': X[:, :, [0, 2]]}, X[:, :, 1:2],
                               run, model, sample_weight=sample_weight, **vars(args))
