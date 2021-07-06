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

def as_minutes(s):
    """
    Converts from seconds to minutes
    :param s: seconds
    :return: string with seconds and minutes
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
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

def get_device(cuda_device):
    """
    Retrieves the torch device associated with GPU `cuda_device`. If CUDA is not available or `cuda_device == -1`,
    then a CPU device is returned
    :param cuda_device: GPU identifier
    :return: Torch device.
    """
    return torch.device("cuda:" + str(cuda_device)
                          if torch.cuda.is_available() and cuda_device != -1
                          else "cpu")