import argparse
from functools import wraps
import json
import os
import shutil
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger, TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger


def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.diff(T, axis=1), np.zeros(T.shape[0])]


def lags_to_times(dT):
    """(N x n_step) matrix of lags -> (N x n_step) matrix of times
    First time is assumed to be zero.
    """
    assert dT.ndim == 2, "dT must be an (N x n_step) matrix"
    return np.c_[np.zeros(dT.shape[0]), np.cumsum(dT[:,:-1], axis=1)]


def parse_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", type=int)
    parser.add_argument("num_layers", nargs="?", type=int)
    parser.add_argument("drop_frac", nargs="?", type=float)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--nb_epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--loss_weights", type=float, nargs='*')
    parser.add_argument("--model_type", type=str, default='lstm')
    parser.add_argument("--decode_type", type=str, default=None)
    parser.add_argument("--gpu_frac", type=float, default=0.31)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=2e-9)
    parser.add_argument("--sim_type", type=str)
    parser.add_argument("--data_type", type=str, default='sinusoid')
    parser.add_argument("--N_train", type=int, default=50000)
    parser.add_argument("--N_test", type=int, default=1000)
    parser.add_argument("--n_min", type=int, default=200)
    parser.add_argument("--n_max", type=int, default=200)
    parser.add_argument('--even', dest='even', action='store_true')
    parser.add_argument('--uneven', dest='even', action='store_false')
    parser.add_argument('--no_train', dest='no_train', action='store_true')
    parser.add_argument("--filter_length", type=int, default=None)
    parser.add_argument('--embedding', type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
    parser.add_argument('--pool', type=int, default=None)
    parser.add_argument("--first_N", type=int, default=None)
    parser.add_argument("--m_max", type=float, default=15.)
    parser.add_argument("--lomb_score", type=float, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--finetune_rate', type=float, default=None)
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser.set_defaults(even=False, batch_norm=False, bidirectional=False)
    args = parser.parse_args()
    if args.model_type in ['conv', 'atrous'] and args.filter_length is None:
        parser.error("--model_type {} requires --filter_length".format(args.model_type))
    return args


def get_run_id(model_type, size, num_layers, lr, drop_frac=0.0, filter_length=None,
               embedding=None, batch_norm=False, pool=None, decode_type=None,
               bidirectional=False, **kwargs):
    run = "{}_{:03d}_x{}_{:1.0e}_drop{}".format(model_type, size,
                                                num_layers, lr,
                                                int(100 * drop_frac)).replace('e-', 'm')
    if batch_norm:
        run += "_bn"
    if filter_length is not None:
        run += '_f{}'.format(filter_length)
    if embedding:
        run += '_emb{}'.format(embedding)
    if pool:
        run += '_pool{}'.format(pool)
    if decode_type:
        run += '_decode{}'.format(decode_type)
    if bidirectional:
        run += '_bidir'

    return run


def limited_memory_session(gpu_frac, gpu_id):
    if gpu_frac <= 0.0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return tf.Session()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) if gpu_id is not None else ''
        gpu_opts = tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac))
        return tf.Session(config=gpu_opts)


def train_and_log(X, Y, run, model, nb_epoch, batch_size, lr, loss, sim_type,
                  metrics=[], sample_weight=None, no_train=False, patience=20,
                  finetune_rate=None, validation_data=None, **kwargs):
    optimizer = Adam(lr=lr if not finetune_rate else finetune_rate)
    print(metrics)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  sample_weight_mode='temporal' if sample_weight is not None else None)

    log_dir = os.path.join(os.getcwd(), 'keras_logs', sim_type, run)
    print(log_dir)
    weights_path = os.path.join(log_dir, 'weights.h5')
    loaded = False
    if os.path.exists(weights_path):
        print("Loading {}...".format(weights_path))
        history = []
        model.load_weights(weights_path)
        loaded = True
    elif no_train or finetune_rate:
        raise FileNotFoundError("No weights found.")

    if finetune_rate:  # write logs to new directory
        log_dir += "_ft{:1.0e}".format(finetune_rate).replace('e-', 'm')
        print(log_dir)

    if not loaded or finetune_rate:
#        shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir)
        param_log = {key: value for key, value in locals().items()}
        param_log.update(kwargs)
        param_log = {k: v for k, v in param_log.items()
                     if k not in ['X', 'Y', 'model', 'optimizer', 'sample_weight',
                                  'kwargs', 'validation_data']}
        json.dump(param_log, open(os.path.join(log_dir, 'param_log.json'), 'w'),
                  sort_keys=True, indent=2)
        history = model.fit(X, Y, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.2,
                            callbacks=[ProgbarLogger(),
                                       TensorBoard(log_dir=log_dir, write_graph=False),
                                       CSVLogger(os.path.join(log_dir, 'training.csv')),
                                       EarlyStopping(patience=patience),
                                       ModelCheckpoint(weights_path, save_weights_only=True)],
                            sample_weight=sample_weight,
                            validation_data=validation_data)
    return history
