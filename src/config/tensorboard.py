import datetime
import os
import threading

from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class MyWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run = None

    def set_run(self, run):
        self.run = run


def create_tensorboard_writer():
    thread_id = threading.get_native_id()
    TENSORBOARD_RUN = datetime.datetime.now().strftime("%m-%d__%H-%M") + f"__{thread_id}"
    TENSORBOARD_LOGDIR = os.path.join(ROOT_DIR,
                                      os.pardir,
                                      os.pardir,
                                      'runs',
                                      TENSORBOARD_RUN)
    writer = MyWriter(log_dir=TENSORBOARD_LOGDIR, flush_secs=30, max_queue=100000)
    writer.run = TENSORBOARD_RUN
    return writer
