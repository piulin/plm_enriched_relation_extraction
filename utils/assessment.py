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
assessment module: Assess the performance of a model.
"""

import sys

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
import os

def plot_confusion_matrix(cm,
                          classes,
                          savefolder,
                          filename = utils.timestamp() + '.png',
                          figsize = (30,20)):
    """
    Creates an image of a confusion matrix.
    :param cm: confusion matrix
    :param classes: lables for each class
    :param savefolder: folder where the image will be saved
    :param filename: name for the image file
    :param figsize: size of the image
    :return: full path to the saved image
    """

    # construct a dataframe from the confusion matrix
    df_cm = pd.DataFrame(cm, index = [i for i in classes],
                  columns = [i for i in classes])

    # create a plot
    plt.figure(figsize = figsize )

    # dye it as a heat map
    sn.heatmap(df_cm, annot=True, fmt='g')

    # save it into a file
    savepath = os.path.join(savefolder, filename)
    plt.savefig(savepath)

    return savepath


def assess(dataset,
           y,
           y_hat,
           log,
           figure_folder,
           label='default',
           plot=True,
           step=None):

    """
    Asses the performance of the model given gold and predicted labels
    :param dataset: dataset that is assessed
    :param y: gold labels
    :param y_hat: predicted labels
    :param log: log system
    :param figure_folder: path to the folder containing figures
    :param label: it is used to name the assessment (e.g. dev or test) in the logs
    :param plot: creates and logs a plot of the confusion matrix
    :param step: x axis value for logs
    :return:
    """


    # check dimensions match
    if ( len(y) != len(y_hat) ):
        print("Cannot assess. Dimension mismatch.", file=sys.stderr)

    # compute confusion matrix from true and predicted labels
    cm = confusion_matrix(y, y_hat)

    # get relation labels from ids
    labels = [dataset.get_relation_label_of_id(i) for i in get_unique_labels(y,y_hat)]

    # create a plot of the confusion matrix
    if plot:
        savepath = plot_confusion_matrix(cm,
                                         labels,
                                         figure_folder,
                                         label + '-step-' + str(0 if step is None else step) + '-' + utils.timestamp() + '.png')
        log.log_artifact(savepath)

    # Retrieve performance metrics

    w_precision, \
    w_recall, \
    w_fscore, \
    _ = precision_recall_fscore_support(y, y_hat, average='weighted', zero_division=0)

    mi_precision, \
    mi_recall, \
    mi_fscore, \
    _ = precision_recall_fscore_support(y, y_hat, average='micro',  zero_division=0)

    ma_precision, \
    ma_recall, \
    ma_fscore, \
    _ = precision_recall_fscore_support(y, y_hat, average='macro',  zero_division=0)

    # log the performance metrics
    print( f' # [{label}] Assessment (weighted):')
    print(f"    @precision: {w_precision * 100:.4f}")
    print(f"    @recall:    {w_recall * 100:.4f}")
    print(f"    @fscore:    {w_fscore * 100:.4f}")

    log.log_metrics(
        {
            'metrics': {
                f' {label} P -weighted': w_precision,
                f' {label} R -weighted': w_recall,
                f' {label} F1 -weighted': w_fscore,
                f' {label} P -micro': mi_precision,
                f' {label} R -micro': mi_recall,
                f' {label} F1 -micro': mi_fscore,
                f' {label} P -macro': ma_precision,
                f' {label} R -macro': ma_recall,
                f' {label} F1 -macro': ma_fscore
            },
            'step': 0 if step is None else step
        }
    )


def get_unique_labels(y, y_hat):
    """
    Get sorted and unique classes from both gold and predicted classes.
    :param y: gold labels
    :param y_hat: predicted labels
    :return: list consisting of sorted and unique classes
    """

    # copy list
    ys = list(y)

    # make one big list
    ys.extend(y_hat)

    # Convert it into numpy array
    ys = np.array(ys)

    # unique
    ys = np.unique(ys)
    # sort them
    ys.sort()

    return list(ys)

