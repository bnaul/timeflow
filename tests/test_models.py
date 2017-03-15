from argparse import Namespace
import tempfile
import sys

from keras_util import parse_model_args
import autoencoder
import period
import period_inverse
import periodogram
import survey_autoencoder


sys.argv = ['']
DEFAULT_ARGS = {"size": 4, "drop_frac": 0.25, "n_min": 4, "n_max": 10,
                "nb_epoch": 1, "N_train": 5, "N_test": 5, "lr": 1e-3,
                "batch_size": 5, "embedding": 2, "filter_length": 3}


def test_period_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    period.main(test_args)


def test_period_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    period.main(test_args)


def test_period_inverse_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    period_inverse.main(test_args)


def test_period_inverse_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    period_inverse.main(test_args)


def test_autoencoder_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    autoencoder.main(test_args)


def test_autoencoder_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = {"num_layers": num_layers, "even": even,
                                 "model_type": model_type, "sim_type": log_dir}
                    test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
                    autoencoder.main(test_args)


def test_survey_autoencoder_rnn():
    for noisify in [False, True]:
        with tempfile.TemporaryDirectory() as log_dir:
            test_args = {"num_layers": 1, "model_type": "gru", "sim_type":
                         log_dir, "noisify": noisify, "survey_files":
                         ["tests/test_lcs.pkl"]}
            test_args = parse_model_args({**DEFAULT_ARGS, **test_args})
            survey_autoencoder.main(test_args)
