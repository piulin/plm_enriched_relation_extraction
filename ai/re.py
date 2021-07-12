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
class re: it is reponsible of performing training and test operations on a given dataset.
"""

import torch.nn as nn
from torch import optim
from ai import enriched_attention_plm
from torch.utils.data import DataLoader
import time
from utils import utils
from log import log
from utils import assessment
from tqdm import tqdm

class re(object):

    def __init__(self,
                 number_of_relations,
                 device,
                 plm_model_path,
                 figure_folder,
                 args):
        """
        Initializes the network
        :param number_of_relations: Number of different relations in the labels
        :param device: device where the computation will take place
        :param plm_model_path: path to the pretrained language model
        :param figure_folder: folder where figures are saved
        :param args: command-line arguments
        """

        # Instantiate the network and send it to the `device`
        self.eat = enriched_attention_plm\
            .enriched_attention_transformers(number_of_relations, plm_model_path)\
            .to(device)

        # Save args locally for future use
        self.device = device
        self.figure_folder = figure_folder

        # Instantiate the log with the command-line arguments
        self.glog = log.log(args)

    def fit(self,
            dataset,
            batch_size,
            learning_rate,
            print_every,
            epochs,
            dev_dataset=None):

        """
        Learns a classifier for relation extraction
        :param dataset: data used to learn the classifier
        :param batch_size: batch size
        :param learning_rate: learning rate for the optimizer
        :param print_every: report loss every `print_every` iterations
        :param epochs: number of epochs for training
        :param dev_dataset: dataset used for development (to report perfomance metrics)
        :return:
        """

        # define loss function. TODO: consider weights for unbalanced data.
        loss_criterion = nn.NLLLoss()

        # define optimizer.
        optimizer = optim.AdamW(lr=learning_rate, params=self.eat.parameters())

        # retrieve the batches for training
        train_iterator = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=dataset.collate,
                                    shuffle=True)


        # set network in training mode (for using e.g. dropout regularization)
        self.eat.train()

        # Iterate epochs
        for i in range(epochs):

            print(f'#####! EPOCH {i + 1} !#####')

            # Train one epoch
            self.train_epoch(train_iterator,
                             optimizer,
                             loss_criterion,
                             print_every,
                             dev_dataset,
                             dataset,
                             batch_size)



    def train_epoch(self,
                    train_iterator,
                    optimizer,
                    loss_criterion,
                    print_every,
                    dev_dataset,
                    train_dataset,
                    batch_size):
        """
        Performs one complete training pass on the dataset
        :param train_iterator: training data splitted into batches
        :param optimizer: optimizer used for updating the parameters
        :param loss_criterion: loss function
        :param print_every: report loss every `print_every` iterations
        :param dev_dataset: dataset used for development (to report performance metrics)
        :param train_dataset: as `dev_dataset`it is used to report performance metrics
        :param batch_size: batch size
        :return:
        """

        # retrieve the number of total iterations of one epoch
        no_iterations = len(train_iterator)

        # Get start timestamp
        start = time.time()

        print_loss = 0.

        # perform a forward and backward pass on the batch
        # Each batch is a tuple (X,y)
        for iter, batch in enumerate(train_iterator, start=1):
            print('.', end='', flush=True)

            # fp and bp
            loss = self.train(batch,
                       optimizer,
                       loss_criterion)

            print_loss += loss

            # log the current loss
            self.glog.log_metrics(
                {
                    'metrics' : {
                        'train_loss': loss
                    },
                    'step': iter
                }
            )

            # Report loss every `print_every` iterations and evaluate on dev and train datasets
            if iter % print_every == 0:

                print('')

                # Evaluate and on train dataset
                self.evaluate(train_dataset, batch_size, 'Train', plot=False)

                # Evaluate on test dataset if provided
                if dev_dataset:
                    self.evaluate(dev_dataset, batch_size, 'Dev', plot=False)

                loss_avg = print_loss / print_every
                print('%s (%d %d%%) loss: %.4f' % (utils.time_since(start, iter / no_iterations),
                                                   iter,
                                                   iter / no_iterations * 100,
                                                   loss_avg,
                                                    ))
                # also log the metric
                self.glog.log_metrics(
                    {
                        'metrics' : {
                            'avg_train_loss': loss_avg
                        },
                        'step': iter
                    }
                )
                print_loss = 0.


    def train(self,
              batch,
              optimizer,
              loss_criterion):
        """
        Performs a forward and backwards pass on the batch
        :param batch: tuple (X,y)
        :param optimizer: optimizer used for updating the parameters
        :param loss_criterion: loss function
        :return: loss
        """


        # Set gradients to zero for mini-batching training
        optimizer.zero_grad()

        # Retrieve X and y from the batch tuple
        X = batch [0]
        y = batch [1]

        # Perform a forward pass
        output = self.eat(X)

        # for i in range(batch_size):
        loss = loss_criterion(output, y)

        # update network parameters
        loss.backward()
        optimizer.step()

        # report loss
        return loss.item()


    def evaluate(self,
                dataset,
                batch_size,
                evaluate_label = 'default',
                plot = True
                ):
        """
        Retrieve the gold and predicted labels of a dataset.
        :param dataset: input data
        :param batch_size: tuple (X,y)
        :param evaluate_label: label for reporting performance into mlflow
        :param plot: decides whether to save the confusion matrix as a heat map
        :return: gold and predcited labels (as lists)
        """

        # Lists for storing true and predicted relations, respectively
        ys_gt = []
        ys_hat = []

        # Split the dataset into batches
        batch_iterator = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=dataset.collate,
                                    shuffle=False)

        # Inform the network that we are going to evaluate on it
        self.eat.eval()

        # iterate batches
        for iter, batch in enumerate(batch_iterator, start=1):
            print('.', end='', flush=True)

            # get ground truth
            y = batch [1]

            # get predicted labels for batch
            y_hat = self.predict(batch)

            # Append gold and predicted labels to lists
            ys_hat.extend(
                y_hat.transpose(0,1).squeeze().tolist()
            )
            ys_gt.extend(
                y.tolist()
            )

            break

        # Assess
        assessment.assess(dataset, ys_gt, ys_hat, self.glog, self.figure_folder, evaluate_label, plot=plot)

        return ys_gt, ys_hat

    def predict(self,
                batch):
        """
        Retrieves the predicted labels from the X data of a batch
        :param batch: tuple (X,y)
        :return: predicted labels (y_hat)
        """


        # Retrieve X and y from the batch tuple
        X = batch [0]

        # Perform a forward pass
        output = self.eat(X)

        # Retrieve the index of that element with highest log probability from the softmax classification
        y_hat = output.topk(1)[1]

        return y_hat