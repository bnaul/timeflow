import numpy as np
from keras import backend as K
from keras.layers import (Input, Dense, TimeDistributed, Activation, LSTM, GRU,
                          Dropout, merge, Reshape, Flatten, RepeatVector,
                          Conv1D, MaxPooling1D)
from keras.models import Model, Sequential

import sample_data
import keras_util as ku

# TODO additive noise (corruption)?
def even_gru_autoencoder(output_len, n_step, size, num_layers, drop_frac, **kwargs):
    main_input = Input(shape=(n_step, output_len))
    encode = [main_input] + [None] * num_layers
    drop_in = [None] * (num_layers + 1)
    for i in range(1, num_layers + 1):
        encode[i] = GRU(size, return_sequences=(i != num_layers))(encode[i - 1])
        drop_in[i] = Dropout(drop_frac)(encode[i])
    tiled = RepeatVector(n_step)(encode[-1])
#    aux_input = Input(shape=(n_step, 1), name='aux_input')
#    decode = merge([aux_input, tiled], mode='concat')
    decode = [tiled] + [None] * num_layers
    drop_out = [None] * (num_layers + 1)
    for i in range(1, num_layers + 1):
        decode[i] = GRU(size, return_sequences=True)(decode[i - 1])
        drop_out[i] = Dropout(drop_frac)(decode[i])
    out_relu = TimeDistributed(Dense(output_len, activation='relu'))(drop_out[-1])
    out_lin = TimeDistributed(Dense(output_len, activation='linear'))(out_relu)
    model = Model(input=main_input, output=out_lin)
    return model


# input: (t, m, e) or (t, m)
# aux input: (t) or (t, e)
# output: just m (output_len==1)
def uneven_gru_autoencoder(input_len, aux_input_len, n_step, size, num_layers, drop_frac, **kwargs):
    output_len = 1

    main_input = Input(shape=(n_step, input_len), name='main_input')
    encode = [main_input] + [None] * num_layers
    drop_in = [None] * (num_layers + 1)
    for i in range(1, num_layers + 1):
        encode[i] = GRU(size, return_sequences=(i != num_layers))(encode[i - 1])
        drop_in[i] = Dropout(drop_frac)(encode[i])
    tiled = RepeatVector(n_step)(encode[-1])
    aux_input = Input(shape=(n_step, aux_input_len), name='aux_input')
    merged = merge([aux_input, tiled], mode='concat')
    decode = [merged] + [None] * num_layers
    drop_out = [None] * (num_layers + 1)
    for i in range(1, num_layers + 1):
        decode[i] = GRU(size, return_sequences=True)(decode[i - 1])
        drop_out[i] = Dropout(drop_frac)(decode[i])
    out_relu = TimeDistributed(Dense(output_len, activation='relu'))(drop_out[-1])
    out_lin = TimeDistributed(Dense(output_len, activation='linear'))(out_relu)
    model = Model(input=[main_input, aux_input], output=out_lin)
    return model


if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    from keras import backend as K

    import sample_data
    import keras_util as ku

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("drop_frac", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str, default='autoencoder')
    parser.add_argument("--filter", type=int, default=5)
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=250)
    parser.add_argument("--n_max", type=int, default=250)
    parser.add_argument('--even', dest='even', action='store_true')
    parser.add_argument('--uneven', dest='even', action='store_false')
    parser.set_defaults(even=True)
    args = parser.parse_args()

    np.random.seed(0)
    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    args.n_min = 250; args.n_max = 250
    X, Y = sample_data.periodic(args.N_train + args.N_test, args.n_min,
                                args.n_max, t_max=2*np.pi, even=True,
                                A_shape=5., noise_sigma=args.sigma, w_min=0.1,
                                w_max=1.)
    X = X[:, :, 1:2]

    model_dict = {'gru': even_gru_autoencoder}

    K.set_session(ku.limited_memory_session(args.gpu_frac, args.gpu_id))
    model = model_dict[args.model_type](output_len=X.shape[-1], n_step=args.n_max,
                                        **vars(args))

    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(args.model_type, args.size,
                                                args.num_layers, args.lr,
                                                int(100 * args.drop_frac)).replace('e-', 'm')
    if 'conv' in run:
        run += '_f{}'.format(args.filter)
    sample_weight = (X[:, :, -1] != -1.)
    history = ku.train_and_log(X[train], X[train], run, model,
			       sample_weight=sample_weight, **vars(args))
