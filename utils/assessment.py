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
from collections import Counter

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

    # Remove no relation TPs
    y, y_hat = remove_no_relation(dataset, y, y_hat)

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

def remove_no_relation(dataset, y, y_hat):
    """
    Removes no relation TP from the gold standard and the predicted labels
    :param y: gold standard
    :param y_hat: predicted labels
    :return: gold standard and predicted labels without no relation TPs.
    """

    # retrieve the no relation ID
    no_relation_id = dataset.no_relation_label()


    # Define output lists
    ny = []
    ny_hat = []


    # Loop over the gold standard and predicted labels
    for i in range(len(y)):

        # if gold and predicted labels are no relation, then we skip them
        if y[i] == no_relation_id and y_hat[i] == no_relation_id:
            continue

        # otherwise, we append them to the new lists
        else:
            ny.append(y[i])
            ny_hat.append(y_hat[i])

    return ny, ny_hat