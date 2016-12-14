from argparse import Namespace
import tempfile

import autoencoder
import period
import period_inverse
import periodogram


DEFAULT_ARGS = {"size": 4, "drop_frac": 0.25, "n_min": 4, "n_max": 10,
                "nb_epoch": 1, "N_train": 5, "N_test": 5, "sigma": 0.,
                "loss_weights": None, "gpu_frac": 0.0, "gpu_id": None,
                "lr": 1e-3, "batch_size": 5, "loss": "mse", "embedding": 1,
                "filter_length": 3}


def test_period_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    period.main(Namespace(**test_args))


def test_period_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    period.main(Namespace(**test_args))


def test_period_inverse_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    period_inverse.main(Namespace(**test_args))


def test_period_inverse_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    period_inverse.main(Namespace(**test_args))


def test_periodogram_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    periodogram.main(Namespace(**test_args))


def test_periodogram_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    periodogram.main(Namespace(**test_args))


def test_autoencoder_conv():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["conv", "atrous"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    autoencoder.main(Namespace(**test_args))


def test_autoencoder_rnn():
    for num_layers in [1, 2]:
        for even in [True, False]:
            for model_type in ["gru"]:
                with tempfile.TemporaryDirectory() as log_dir:
                    test_args = DEFAULT_ARGS.copy()
                    test_args.update({"num_layers": num_layers, "even": even,
                                      "model_type": model_type,
                                      "sim_type": log_dir})
                    autoencoder.main(Namespace(**test_args))
