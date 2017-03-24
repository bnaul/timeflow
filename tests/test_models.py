from argparse import Namespace
import sys

from keras_util import parse_model_args
import autoencoder
import period
import period_inverse
import periodogram
import survey_autoencoder
import survey_classifier


sys.argv = ['']
DEFAULT_ARGS = {"size": 4, "drop_frac": 0.25, "n_min": 4, "n_max": 10,
                "nb_epoch": 1, "N_train": 5, "N_test": 5, "lr": 1e-3,
                "batch_size": 5, "embedding": 2, "filter_length": 3}


def test_period_conv(tmpdir):
    for num_layers in [1, 2]:
        for model_type in ["conv", "atrous"]:
            test_args = {"num_layers": num_layers, "model_type": model_type,
                         "sim_type": str(tmpdir) + "_period",
                         **DEFAULT_ARGS}
            period.main(test_args)


def test_period_rnn(tmpdir):
    for num_layers in [1, 2]:
        for model_type in ["gru"]:
            test_args = {"num_layers": num_layers, "model_type": model_type,
                         "sim_type": str(tmpdir) + "_period",
                         **DEFAULT_ARGS}
            period.main(test_args)


def test_autoencoder_conv(tmpdir):
    for num_layers in [1, 2]:
        for model_type in ["conv", "atrous"]:
            test_args = {"num_layers": num_layers, "model_type": model_type,
                         "sim_type": str(tmpdir) + "_auto",
                         **DEFAULT_ARGS}
            autoencoder.main(test_args)


def test_autoencoder_rnn(tmpdir):
    for num_layers in [1, 2]:
        for model_type in ["gru"]:
            test_args = {"num_layers": num_layers, "model_type": model_type,
                         "sim_type": str(tmpdir) + "_auto",
                         **DEFAULT_ARGS}
            autoencoder.main(test_args)


def test_survey_autoencoder_rnn(tmpdir):
    for noisify in [False, True]:
        test_args = {"num_layers": 1, "model_type": "gru",
                     "noisify": noisify, "survey_files": ["tests/test_lcs.pkl"],
                     "sim_type": str(tmpdir) + "_survey_auto" + ("_noise" if noisify else ""),
                     **DEFAULT_ARGS}
        survey_autoencoder.main(test_args)


def test_survey_classifier_rnn(tmpdir):
    test_args = {"num_layers": 1, "model_type": "gru",
                 "survey_files": ["tests/test_lcs.pkl"],
                 "sim_type": str(tmpdir) + "_survey_classifier",
                 **DEFAULT_ARGS}
    survey_classifier.main(test_args)
