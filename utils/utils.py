import os
import random
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import logging
logger = logging.getLogger('main_all')

def get_optimizer(cfg, model):
    optimizer_cls = cfg["Optimizer"]["Name"]
    optimizer_args = {k: v for k, v in cfg["Optimizer"].items() if k != "Name"}
    return eval(f"torch.optim.{optimizer_cls}")(
        model.parameters(), **optimizer_args
    )


def get_criterion(cfg):
    criterion_cls = cfg["Loss"]["Name"]
    criterion_args = {k: v for k, v in cfg["Loss"].items() if k != "Name"}
    return eval(f"torch.nn.{criterion_cls}")(**criterion_args)


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = logdir / run_name
        if not log_path.exists():
            log_path.mkdir(parents=True)
            return log_path
        i = i + 1


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """ function to get the device """ 
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProgressParallel(Parallel):
    def __init__(
        self, use_tqdm=True, total=None, tqdm_params=None, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._tqdm_params = tqdm_params if tqdm_params is not None else {}
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._use_tqdm:
            with tqdm(total=self._total, **self._tqdm_params) as self._pbar:
                result = super().__call__(*args, **kwargs)
                self._pbar.close()  # Ensure closure of the tqdm bar
                return result
        else:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._pbar:
            if self._total is None:
                self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

def get_scalars_from_tensorboard(summary_dir=None, date=None, time=None):
    """
    Read data from the TensorBoard summary writer and return them as structured.

    Args:
        summary_dir (str or Path, optional): Path to the summary writer directory.
        date (str or int, optional): Date for the log.
        time (str, optional): Time for the log.

    Returns:
        dict: A dictionary with keys as the prettified scalar names and values as tuples of (list of steps, list of values).

    Raises:
        ValueError: If required parameters are not provided or the summary directory does not exist.
    """
    if summary_dir is None:
        if date is not None and time is not None:
            base_dir = Path(__file__).resolve().parents[2].absolute()
            summary_dir = base_dir / "logs" / str(date) / time / 'summary_writer'
        else:
            raise ValueError("Either 'summary_dir' or both 'date' and 'time' must be provided.")
    summary_dir = Path(summary_dir)

    if not summary_dir.is_dir():
        raise ValueError(f"The folder '{summary_dir}' does not exist.")

    event_accumulator = EventAccumulator(str(summary_dir)).Reload()
    scalars = event_accumulator.Tags()['scalars']

    replace_table = {'/': ': ', 'val': 'validation', 'acc': 'accuracy', '_': ' '}
    results = {}
    for scalar in scalars:
        steps, values = zip(*[(s.step, s.value) for s in event_accumulator.Scalars(scalar)])
        scalar_name = ''.join([replace_table.get(char, char) for char in scalar])
        results[scalar_name] = (list(steps), list(values))

    return results

if __name__ == "__main__":

    # Mock-up model to use with the optimizer and loss function
    import sys 
    import yaml 
    import torch.nn as nn
    
    with open(sys.argv[1]) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Getting optimizer and criterion using config
    optimizer = get_optimizer(cfg, model)
    criterion = get_criterion(cfg)

    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    # Example usage of the generate_unique_logpath function
    logdir = Path("./logs")
    unique_logpath = generate_unique_logpath(logdir, "test_run")
    print(f"Unique log path created at: {unique_logpath}")

    # Seed setting example
    seed_everything(123)

    # ProgressParallel demonstration with a simple parallel task
    from joblib import delayed
    def sample_function(x):
        return x * x

    parallel_runner = ProgressParallel(n_jobs=2, total=10, tqdm_params={'desc': 'Squaring numbers'})
    results = parallel_runner(delayed(sample_function)(i) for i in range(10))
    print(f"Parallel processing results: {results}")

    # Assuming summary data is available at this path for demonstration
    # Please ensure to have a TensorBoard summary directory to actually run this part
    summary_dir = Path("./logs/some_date/some_time/summary_writer")
    try:
        scalars = get_scalars_from_tensorboard(summary_dir)
        print(f"Scalars retrieved from TensorBoard: {scalars}")
    except ValueError as e:
        print(e)
