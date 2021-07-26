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
from pandas import DataFrame

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
from collections import Counter
from datasets.dataset import dataset
from log.log import log
from typing import List, Tuple
from numpy import ndarray


def plot_confusion_matrix(cm: ndarray,
                          classes: List[str],
                          savefolder: str,
                          filename: str = utils.timestamp() + '.png',
                          figsize: Tuple[int, int] = (30,20)) -> str:
    """
    Creates an image of a confusion matrix.
    :param cm: confusion matrix of shape (n_classes, n_classes)
    :param classes: string labels for each class of size n_classes
    :param savefolder: folder where the image will be saved
    :param filename: name for the image file
    :param figsize: size of the image
    :return: path to the saved image
    """

    # construct a dataframe from the confusion matrix
    df_cm: DataFrame = pd.DataFrame(cm, index = [i for i in classes],
                  columns = [i for i in classes])

    # create a plot
    plt.figure( figsize = figsize )

    # dye it as a heat map
    sn.heatmap(df_cm, annot=True, fmt='g')

    # save it into a file
    savepath: str = os.path.join(savefolder, filename)
    plt.savefig(savepath)

    return savepath


def assess(dataset: dataset,
           y: List[int],
           y_hat: List[int],
           log: log,
           figure_folder: str,
           label: str = 'default',
           plot: bool = True,
           step: int = None) -> None:

    """
    Asses the performance of the model given gold and predicted labels
    :param dataset: dataset to be assessed
    :param y: gold labels (size n)
    :param y_hat: predicted labels (same size n)
    :param log: log system, to post performance metrics
    :param figure_folder: save figures in this folder if `plot` is `True`
    :param label: Name the assessment (e.g. dev or test) in the logs
    :param plot: saves the confusion matrix as a png figure into the `figure_folder` folder
    :param step: optional x axis value used in the log system
    :return:
    """


    # check dimensions match
    assert ( len(y) == len(y_hat) )

    # Remove no relation TPs
    y, y_hat = remove_no_relation(dataset, y, y_hat)

    # compute confusion matrix from true and predicted labels
    cm: ndarray = confusion_matrix(y, y_hat) # cm[n_classes,n_classes]

    # get relation labels from ids
    labels: List[str] = [dataset.get_relation_label_of_id(i) for i in get_unique_labels(y,y_hat)]

    # create a plot of the confusion matrix
    if plot:
        savepath: str = plot_confusion_matrix(cm,
                                         labels,
                                         figure_folder,
                                         label + '-step-' + str(0 if step is None else step) + '-' + utils.timestamp() + '.png')
        log.log_artifact(savepath)

    # Retrieve performance metrics
    w_precision: float
    w_recall: float
    w_fscore: float

    w_precision, \
    w_recall, \
    w_fscore, \
    _ = precision_recall_fscore_support(y, y_hat, average='weighted', zero_division=0)

    mi_precision: float
    mi_recall: float
    mi_fscore: float

    mi_precision, \
    mi_recall, \
    mi_fscore, \
    _ = precision_recall_fscore_support(y, y_hat, average='micro',  zero_division=0)

    ma_precision: float
    ma_recall: float
    ma_fscore: float

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

# from https://github.com/yuhaozhang/tacred-relation/blob/master/utils/scorer.py
def score(key, prediction, NO_RELATION, log, label, step = None, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))


    log.log_metrics(
        {
            'metrics': {
                f' {label} P -TCMODULE': prec_micro,
                f' {label} R -TCMODULE': recall_micro,
                f' {label} F1 -TCMODULE': f1_micro,
            },
            'step': 0 if step is None else step
        }
    )

    return prec_micro, recall_micro, f1_micro

def get_unique_labels(y: List[int],
                      y_hat: List[int]) -> List[int]:
    """
    Get sorted and unique classes from both gold and predicted classes.
    :param y: gold labels (size n)
    :param y_hat: predicted labels (also size n)
    :return: list consisting of sorted and unique classes
    """

    # copy list
    ys: List[int] = list(y)

    # make one big list
    ys.extend(y_hat)

    # Convert it into a numpy array
    ys: ndarray = np.array(ys) # ys[n]

    # unique
    ys = np.unique(ys)
    # sort them
    ys.sort()

    return list(ys)

def remove_no_relation( dataset: dataset,
                       y: List[int],
                       y_hat: List[int] ) -> Tuple[List[int], List[int]]:
    """
    Removes no-relation TP (true positive) from the gold standard and the predicted labels
    :param y: gold standard
    :param y_hat: predicted labels
    :return: gold standard and predicted labels without no-relation TPs.
    """

    # retrieve the no relation ID
    no_relation_id: int = dataset.no_relation_label()


    # Define output lists
    ny: List[int] = []
    ny_hat: List[int] = []


    # Loop over the gold standard and predicted labels
    i: int
    for i in range(len(y)):

        # if gold and predicted labels are no relation, then we skip them
        if y[i] == no_relation_id and y_hat[i] == no_relation_id:
            continue

        # otherwise, we append them to the new lists
        else:
            ny.append(y[i])
            ny_hat.append(y_hat[i])

    return ny, ny_hat