"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM
    
    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),
    
    with the supervision of
    
    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""


"""
utils module: it gathers a bunch of utilities that provide side functionality.
"""


import time
import math
import torch
from datetime import datetime
import os
from log.teletype import teletype

def as_minutes(s: float) -> str:
    """
    Converts from seconds to minutes
    :param s: seconds
    :return: seconds and minutes formatted in a string
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since: float,
               percent: float) -> str:
    """
    Retrieves the elapsed time between two moments in seconds, and the remaining time for ending the task
    :param since: start timestamp
    :param percent: percentage [0,1] of the task completion
    :return: elapsed time formatted in a nice string
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def get_device(cuda_device: int) -> torch.device:
    """
    Retrieves the torch device associated with GPU `cuda_device`. If CUDA is not available or `cuda_device == -1`,
    then a CPU device is returned
    :param cuda_device: GPU identifier
    :return: Torch device.
    """

    # print information for the user
    teletype.start_task(f'Configuring cuda device: {cuda_device}', __name__)

    if not torch.cuda.is_available() and cuda_device != -1:
        teletype.finish_task(__name__, success=False, message='Using CPU')

    elif torch.cuda.is_available() and cuda_device != -1:
        teletype.finish_task(__name__)
    else:
        teletype.finish_task(__name__, message='Using CPU')

    # retrieve device
    return torch.device("cuda:" + str(cuda_device)
                          if torch.cuda.is_available() and cuda_device != -1
                          else "cpu")

def timestamp() -> str :
    """
    Retrieves a string corresponding to the current timestamp
    :return:
    """

    now: datetime = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

def create_folder(path: str) -> None:
    """
    Creates a folder given a `path` if there is no directory of the same name
    :param path:
    :return:
    """

    teletype.start_task(f'Creating folder: "{path}"', __name__)

    try:
        os.makedirs(path)
        teletype.finish_task(__name__)
    except FileExistsError:
        teletype.finish_task(__name__, success=False, message='Folder already exists')
        pass