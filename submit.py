from argparse import Namespace
from multiprocessing import Pool, current_process
import sys
import tempfile
from sklearn.model_selection import ParameterGrid
from keras_util import parse_model_args, limited_memory_session
from autoencoder import main as autoencoder
from period import main as period
from asas import main as asas
from asas_full import main as asas_full


def set_session(num_gpus, procs_per_gpu, tmpdirname):
    gpu_frac = 0.96 / procs_per_gpu  # can get OOM error if you use exactly 100%
    gpu_id = int(current_process().name.split('-')[-1]) % num_gpus
    limited_memory_session(gpu_frac, gpu_id)
    log_file = tempfile.NamedTemporaryFile('w')
    print(log_file.name)
    sys.stdout = log_file


if __name__ == '__main__':
    NUM_GPUS = 2
    PROCS_PER_GPU = 5

    simulation = period
    model_types = ['conv', 'atrous']
    params = {
        'sim_type': ['period/uneven/noise0.0'],
        'size': [32, 64, 96],
        'num_layers': [2, 3, 4],
        'drop_frac': [0.25],
        'n_min': [200], 'n_max': [200], 'sigma': [0.0], 
        'nb_epoch': [500], 'lr': [5e-4], 'patience': [20]
    }
    conv_only_args = {
        'filter_length': [7], 'batch_norm': [True]
    }
    param_grid = ParameterGrid([{'model_type': [t], **params,
                                 **(conv_only_args if t in ['conv', 'atrous'] else {})}
                                for t in model_types])
    full_param_grids = [parse_model_args(p) for p in param_grid]

    with tempfile.TemporaryDirectory() as tmpdirname:
        with Pool(NUM_GPUS * PROCS_PER_GPU, initializer=set_session,
                  initargs=(NUM_GPUS, PROCS_PER_GPU, tmpdirname)) as pool:
            r = pool.map_async(simulation, full_param_grids, chunksize=1)
            r.wait()
