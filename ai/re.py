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
from typing import List, Union, Any, Tuple

from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
import torch

from ai.schemas.ESS_plm import ess_plm

"""
class re: it is responsible of performing training and test on a given a dataset and a classification schema 
for relation extraction.
"""

import torch.nn as nn
from torch import optim, Tensor
from ai.schemas import ESS_plm
from ai.schemas.cls_plm import cls_plm
from ai.schemas.enriched_attention_plm import enriched_attention_plm
from torch.utils.data import DataLoader
import time
from utils import utils
from log import log
from utils import assessment
from datasets.dataset import dataset
from transformers import BatchEncoding


class re(object):

    def __init__(self,
                 **kwargs: dict
                 ):
        """
        Initializes the network

        :param **kwargs: See below

        :Keyword Arguments:
        :param number_of_relations: Number of different relations in the labels
        :param device: device where the computation will take place
        :param plm_model_path: path to the pretrained language model
        :param figure_folder: folder where figures are saved
        :param vocabulary_length: the length of the vocabulary, i.e. the length of the tokenizer.
        :param num_position_embeddings: number of different position embeddings (look-up table size)
        :param position_embedding_size: position embedding size
        :param num_dependency_distance_embeddings: number of different dependency distance embeddings
        :param dependency_distance_size: size of the dependency distance embeddings
        :param attention_size: dimension of the internal attention space (A)
        """

        # Save args locally for future use
        self.device: torch.device = kwargs.get('device', torch.device('cpu'))
        self.figure_folder: str = kwargs.get('figure_folder')
        self.schema: str = kwargs.get('schema')

        # Instantiate the network
        self.eat: Union[ess_plm, cls_plm, enriched_attention_plm] = self.load_network(**kwargs)

        # Instantiate the log with the model's arguments
        self.glog: log.log = log.log(kwargs)

    def load_network(self,
                     **kwargs: dict
                     ) -> Union[ess_plm, cls_plm, enriched_attention_plm]:
        """
        Loads the network given a choice schema

        :param kwargs: See below

        :Keyword Arguments:

        :param number_of_relations: Number of different relations in the labels
        :param vocabulary_length: the length of the vocabulary, i.e. the length of the tokenizer.
        :param plm_model_path: path to the pretrained language model
        :param device: device where the computation will take place

        :param num_position_embeddings: number of different position embeddings (look-up table size)
                                        (Only for Enriched Attention schema)
        :param position_embedding_size: position embedding size (Only for Enriched Attention schema)
        :param num_dependency_distance_embeddings: number of different dependency distance embeddings
            (Only for Enriched Attention schema)
        :param dependency_distance_size: size of the dependency distance embeddings (Only for Enriched Attention schema)
        :param attention_size: dimension of the internal attention space (A) (Only for Enriched Attention schema)
        :return: network
        """

        # switch schema
        if self.schema == 'ESS':
            return ESS_plm \
                .ess_plm(**kwargs) \
                .to(self.device)

        elif self.schema == 'Standard':
            return cls_plm(**kwargs).to(self.device)

        elif self.schema == 'Enriched_Attention':
            return enriched_attention_plm(**kwargs).to(self.device)

    def fit(self,
            dataset: dataset,
            batch_size: int,
            learning_rate: List[float],
            print_every: int,
            epochs: int,
            dev_dataset: dataset = None) -> None:
        """
        Learns a classifier for relation extraction
        :param dataset: dataset used to learn the classifier
        :param batch_size: batch size
        :param learning_rate: learning rate for the optimizers. learning_rate[0] targets the post
        transformer layers (PTLs) whereas learning_rate[1] targets the PLM.
        :param print_every: report loss and performance metric every `print_every` batches in an epoch
        :param epochs: number of epochs for training
        :param dev_dataset: dataset used for development (to report performance metrics)
        :return:
        """

        # define loss function. TODO: consider weights for unbalanced data.
        loss_criterion: NLLLoss = nn.NLLLoss()

        # define optimizer with different learning rates for the plm and the PTLs.
        optimizer: Adam = optim.Adam(
            [{
                'params': self.eat.post_plm_parameters,
                'lr': learning_rate[0]  # PTL
            },
                {
                    'params': self.eat.plm_parameters,
                    'lr': learning_rate[1]  # 'PLM'
                }]
        )

        # retrieve the batches for training
        train_iterator: DataLoader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=dataset.collate,
                                    shuffle=True)

        # define scheduler (from: https://huggingface.co/transformers/training.html)
        num_training_steps: int = epochs * len(train_iterator)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # set network in training mode (for using e.g. dropout regularization)
        self.eat.train()

        # Ctrl-C (SIGINT) signals: stop the training process and proceed with testing
        try:

            # Iterate epochs
            i: int
            for i in range(epochs):

                print('')
                print(f'#####! EPOCH {i + 1} !#####')
                print('')

                # Train one epoch
                self.train_epoch(train_iterator,
                                 optimizer,
                                 lr_scheduler,
                                 loss_criterion,
                                 print_every,
                                 dev_dataset,
                                 dataset,
                                 batch_size)

                # At the end of one epoch, evaluate on the whole dev dataset
                if dev_dataset:
                    print('')
                    print('####! EVAL ON DEV DATASET !####')
                    print('')
                    self.evaluate(dev_dataset, batch_size, f'Dev Epoch', no_batches=None, plot=True, step=i + 1)

        # if Ctrl-C smashed, stop the training process
        except KeyboardInterrupt:

            # Log the early stop
            self.glog.log_param('Early stop at epoch', i + 1)
            print('TRAINING STOPPED.')

    def train_epoch(self,
                    train_iterator: DataLoader,
                    optimizer: Any,
                    lr_scheduler: Any,
                    loss_criterion: Any,
                    print_every: int,
                    dev_dataset: Union[dataset, None],
                    train_dataset: dataset,
                    batch_size: int) -> None:
        """
        Performs one complete training pass on the dataset
        :param train_iterator: training data split into batches
        :param optimizer: optimizer used for updating the parameters
        :param lr_scheduler: learning rate scheduler for learning rate decay
        :param loss_criterion: loss function
        :param print_every: report loss every `print_every` iterations
        :param dev_dataset: dataset used for development (to report performance metrics)
        :param train_dataset: as `dev_dataset`it is used to report performance metrics
        :param batch_size: batch size
        :return:
        """

        # retrieve the number of total iterations of one epoch
        no_iterations: int = len(train_iterator)

        # Get start timestamp
        start: float = time.time()

        print_loss: float = 0.

        # perform a forward and backward pass on the batch
        # Each batch is a tuple (X,y)
        iter: int
        batch: Tuple
        for iter, batch in enumerate(train_iterator, start=1):
            print('.', end='', flush=True)

            # fp and bp
            loss: float = self.train(batch,
                              optimizer,
                              lr_scheduler,
                              loss_criterion)

            print_loss += loss

            # log the current loss
            self.glog.log_metrics(
                {
                    'metrics': {
                        'train_loss': loss
                    },
                    'step': iter
                }
            )

            # Report loss every `print_every` iterations and evaluate on dev and train datasets
            if iter % print_every == 0:

                print('')

                # Evaluate and on train dataset
                self.evaluate(train_dataset, batch_size, 'Train', no_batches=10, plot=False, step=iter)

                # Evaluate on test dataset if provided
                if dev_dataset:
                    self.evaluate(dev_dataset, batch_size, 'Dev', no_batches=10, plot=False, step=iter)

                loss_avg: float = print_loss / print_every
                print('%s (%d %d%%) loss: %.4f' % (utils.time_since(start, iter / no_iterations),
                                                   iter,
                                                   iter / no_iterations * 100,
                                                   loss_avg,
                                                   ))
                # also log the metric
                self.glog.log_metrics(
                    {
                        'metrics': {
                            'avg_train_loss': loss_avg
                        },
                        'step': iter
                    }
                )
                print_loss = 0.

    def train(self,
              batch: Tuple,
              optimizer: Any,
              lr_scheduler: Any,
              loss_criterion: Any) -> float:
        """
        Performs a forward and backwards pass on the batch
        :param batch: n-tuple (X,y,...), where ... can be any other data necessary for training, e.g. entity indices.
        :param optimizer: optimizer used for updating the parameters
        :param lr_scheduler: learning rate scheduler for learning rate decay
        :param loss_criterion: loss function
        :return: loss
        """

        # Set gradients to zero for mini-batching training
        optimizer.zero_grad()

        # Perform a forward pass
        output: Tensor # output[batch_size, n_classes]
        y: Tensor # y[batch_size]
        output, y = self.perform_forward_pass(batch)

        loss: Tensor = loss_criterion(output, y) # loss[1]

        # update network parameters
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # report loss
        return loss.item()

    def perform_forward_pass(self,
                             batch: Tuple) -> Tuple[Tensor, Tensor]:
        """
        Performs a forward pass using the underlying network
        :param batch: training batch tuple
        :return: output of the network (shape [batch_size, n_classes]) and gold labels (shape [batch_size])
        """

        # Retrieve X and y from the batch tuple
        X: BatchEncoding = batch[0]
        y: Tensor = batch[1] # y[batch_size]

        # Switch schema and perform forward task depending on the underlying network
        if self.schema == 'ESS':

            #  indices to the first and second start ETM
            e1_indices: List[int] = batch[2] # of size `batch_size`
            e2_indices: List[int] = batch[3] # of size `batch_size`

            return self.eat(X,
                            e1_indices,
                            e2_indices), y

        elif self.schema == 'Standard':
            return self.eat(X), y

        elif self.schema == 'Enriched_Attention':

            # retrieve features necessary for enriched attention from the batch
            de1: Tensor # de1[batch_size, padded_sentence_length]
            de2: Tensor # de2[batch_size, padded_sentence_length]
            sdp_flag: Tensor # sdp_flag[batch_size, padded_sentence_length]
            sdp: BatchEncoding
            po: Tensor # de1[batch_size, padded_sentence_length]
            ps: Tensor # de1[batch_size, padded_sentence_length]
            de1, de2, sdp_flag, sdp, po, ps = batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

            return self.eat(X, ps, po, de1, de2, sdp_flag, sdp), y

    def evaluate(self,
                 dataset: dataset,
                 batch_size: int,
                 evaluate_label: str = 'default',
                 no_batches: Union[int, None] = 10,
                 plot: bool = True,
                 step: int = None
                 ) -> Tuple[List[int], List[int]]:
        """
        Retrieves the gold and predicted labels of a dataset, and evaluates the performance of the model
        :param dataset: input data
        :param batch_size: batch size
        :param evaluate_label: id label for reporting performance into mlflow
        :param no_batches: number of random batches to be evaluated. If `no_batches` is None, then the evaluation is
                performed on the entire dataset.
        :param plot: decides whether to save the confusion matrix as a heat map
        :param step: x axis value for logs
        :return: gold and predicted labels (as lists of size `batch_size`)
        """

        # Lists for storing true and predicted relations, respectively
        ys_gt: List[int] = []
        ys_hat: List[int] = []

        # Split the dataset into batches
        batch_iterator: DataLoader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=dataset.collate,
                                    shuffle=False if no_batches is None else True)

        # Inform the network that we are going to evaluate on it
        self.eat.eval()

        print('*', end='', flush=True)

        # iterate batches
        iter: int
        batch: Tuple
        for iter, batch in enumerate(batch_iterator, start=1):
            # get ground truth
            y: Tensor = batch[1] # y[batch_size]

            # get predicted labels for batch
            y_hat: Tensor = self.predict(batch) # y_hat[batch_size]

            # Append gold and predicted labels to lists
            ys_hat.extend(
                y_hat.transpose(0, 1).squeeze().tolist()
            )
            ys_gt.extend(
                y.tolist()
            )

            # early stop
            if no_batches is not None and iter > no_batches:
                break

        # Assess
        assessment.assess(dataset, ys_gt, ys_hat, self.glog, self.figure_folder, evaluate_label, plot=plot, step=step)
        assessment.score(ys_gt, ys_hat, dataset.no_relation_label(), self.glog, evaluate_label, step=step)

        return ys_gt, ys_hat

    def predict(self,
                batch: Tuple) -> Tensor:
        """
        Retrieves the predicted labels from the X data of a batch
        :param batch: n-tuple (X,y,...)
        :return: predicted labels (y_hat) of shape [batch_size]
        """

        # Perform a forward pass
        output: Tensor # output[batch_size, n_classes]
        output, _ = self.perform_forward_pass(batch)

        # Retrieve the index of that element with highest log probability from the softmax classification
        y_hat: Tensor = output.topk(1)[1] # y_hat[batch_size]

        return y_hat
