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

class re(object):

    def __init__(self,
                 number_of_relations,
                 device,
                 plm_model_path,
                 args):
        """
        Initializes the network
        :param number_of_relations: Number of different relations in the labels
        :param device: device where the computation will take place
        :param plm_model_path: path to the pretrained language model
        :param args: command-line arguments
        """

        # Instantiate the network and send it to the `device`
        self.eat = enriched_attention_plm\
            .enriched_attention_transformers(number_of_relations, plm_model_path)\
            .to(device)

        self.device = device

        # Instantiate the log with the command-line arguments
        self.glog = log.log(args)

    def fit(self,
            dataset,
            batch_size,
            learning_rate,
            print_every,
            epochs):

        """
        Learns a classifier for relation extraction
        :param dataset: data used to learn the classifier
        :param batch_size: batch size
        :param learning_rate: learning rate for the optimizer
        :param print_every: report loss every `print_every` iterations
        :param epochs: number of epochs for training
        :return:
        """

        # define loss function. TODO: consider weights for unbalanced data.
        loss_criterion = nn.NLLLoss()

        # define optimizer.
        optimizer = optim.Adam(lr=learning_rate, params=[p for p in self.eat.parameters() if p.requires_grad])

        # retrieve the batchs for training
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
                             print_every)



    def train_epoch(self,
                    train_iterator,
                    optimizer,
                    loss_criterion,
                    print_every):
        """
        Performs one complete training pass of the dataset
        :param train_iterator: training data splitted into batches
        :param optimizer: optimizer used for updating the parameters
        :param loss_criterion: loss function
        :param print_every: report loss every `print_every` iterations
        :return:
        """

        # retrieve the number of total iterations of one epoch
        no_iterations = len(train_iterator)

        # Get start timestamp
        start = time.time()

        loss = 0.
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

            print('-', end='', flush=True)

            # log the current loss
            self.glog.log_metrics({
                'train_loss': loss
            },
            iter)

            # Report loss every `print_every` iterations
            if iter % print_every == 0:

                loss_avg = print_loss / print_every
                print('%s (%d %d%%) loss: %.4f' % (utils.time_since(start, iter / no_iterations),
                                                   iter,
                                                   iter / no_iterations * 100,
                                                   loss_avg,
                                                    ))
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
