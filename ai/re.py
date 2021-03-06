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
from typing import List, Union, Any, Tuple, Optional, Dict

from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import  get_scheduler, SchedulerType
import torch

from ai.optimizer.MyAdagrad import MyAdagrad
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
from log.teletype import teletype

class re(object):

    def __init__(self,
                 schema: str,
                 device: torch.device,
                 figure_folder: str,
                 **kwargs: dict
                 ):
        """
        Initializes the classification schema for relation extraction

        :param schema: Selects the classification schema. Choose between `Standard`, `ESS`, or `Enriched_Attention`.
        :param device: device where the computation will take place
        :param figure_folder: folder where figures are saved

        :param **kwargs: parameters needed for the initialization of the schema `schema`

        """

        teletype.start_task(f'Setting up schema "{schema}"', __name__)

        # Save args locally for future use
        self.device: torch.device = device
        self.figure_folder: str = figure_folder
        self.schema: str = schema

        # Instantiate the network (schema-dependent)
        self.eat: Union[ess_plm, cls_plm, enriched_attention_plm] = self.load_network(**kwargs)

        # Instantiate the log with the model's arguments
        self.glog: log.log = log.log(kwargs)

        teletype.finish_task(__name__)

    def load_network(self,
                     **kwargs: dict
                     ) -> Union[ess_plm, cls_plm, enriched_attention_plm]:
        """
        Loads the network given a choice schema
        :param kwargs: parameters to initialize the classification schema
        :return: network
        """

        # switch schema
        if self.schema == 'ess':
            return ESS_plm \
                .ess_plm(**kwargs) \
                .to(self.device)

        elif self.schema == 'standard':
            return cls_plm(**kwargs).to(self.device)

        elif self.schema == 'enriched_attention':
            return enriched_attention_plm(**kwargs).to(self.device)


    def get_optimizer(self,
                      name: str,
                      learning_rate: List[float]) -> torch.optim :
        """
        Retrives a specific optimizer given its `name`
        :param name: name of the optimizer to be used
        :param learning_rate: learning_rate[0] targets the post
        transformer layers (PTLs) whereas learning_rate[1] targets the PLM.
        :return: optimizer
        """

        # define parameters to pass to optimizer.
        # Parameters define the learning rates for each set of them
        parameters: List = [
                    {
                        'params': self.eat.post_plm_parameters,
                        'lr': learning_rate[1]  # PTL
                    },
                    {
                        'params': self.eat.plm_parameters,
                        'lr': learning_rate[0]  # 'PLM'
                    }
                    ]

        # switch optimizer
        if name == 'Adam':
            return optim.Adam( parameters,
                betas=(0.9, 0.99)
            )
        if name == 'AdamW':
            return optim.AdamW( parameters,
                betas=(0.9, 0.99)
            )

        if name == 'SDG':
            return optim.SGD( parameters )

        if name == 'Adamax':
            return optim.Adamax(parameters , betas=(0.9, 0.99) )

        if name == 'MyAdagrad':
            # use my own adagrad to allow for init accumulator value
            return MyAdagrad( parameters, init_accu_value=0.1 )


    def fit(self,
            dataset: dataset,
            batch_size: int,
            learning_rate: List[float],
            print_every: int,
            epochs: int,
            no_eval_batches: int,
            optimizer_name: str,
            scheduler_name: str,
            dev_dataset: dataset = None) -> None:
        """
        Learns a classifier for relation extraction
        :param dataset: dataset used to learn the classifier
        :param batch_size: batch size
        :param learning_rate: learning rate for the optimizers. learning_rate[0] targets the post
        transformer layers (PTLs) whereas learning_rate[1] targets the PLM.
        :param print_every: report loss and performance metric every `print_every` batches in an epoch
        :param epochs: number of epochs for training
        :param no_eval_batches: number of random batches to be assessed every `print_every` iterations.
        :param optimizer_name: name of the optimizer to be used
        :param scheduler_name: name of the scheduler to be used
        :param dev_dataset: dataset used for development (to report performance metrics)
        :return:
        """

        teletype.start_task('Training', __name__)

        # define loss function. TODO: consider weights for unbalanced data.
        loss_criterion: NLLLoss = nn.NLLLoss()

        # define optimizer with different learning rates for the plm and the PTLs.
        optimizer: torch.optim = self.get_optimizer(optimizer_name, learning_rate)

        # retrieve the batches for training
        train_iterator: DataLoader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    collate_fn=dataset.collate,
                                    shuffle=True)

        # define scheduler (from: https://huggingface.co/transformers/training.html)
        num_training_steps: int = epochs * len(train_iterator)

        lr_scheduler = get_scheduler(
            scheduler_name,
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

                teletype.start_subtask(f'Epoch {i+1}', __name__, 'fit')

                # Train one epoch
                self.train_epoch(train_iterator,
                                 optimizer,
                                 lr_scheduler,
                                 loss_criterion,
                                 print_every,
                                 dev_dataset,
                                 dataset,
                                 batch_size,
                                 no_eval_batches)

                # At the end of one epoch, evaluate on the whole dev dataset
                if dev_dataset:

                    self.evaluate(dev_dataset, batch_size, f'Dev Epoch', no_batches=None, plot=True, step=i + 1)

                teletype.finish_subtask(__name__, 'fit')

        # if Ctrl-C smashed, stop the training process
        except KeyboardInterrupt:

            # Log the early stop
            self.glog.log_param('Early stop at epoch', i + 1)
            teletype.finish_subtask(__name__, 'fit', success=False)
            teletype.finish_task(__name__, message=f'Early stop at epoch {i+1} (SIGINT)')
            return

        teletype.finish_task(__name__)

    def train_epoch(self,
                    train_iterator: DataLoader,
                    optimizer: Any,
                    lr_scheduler: Any,
                    loss_criterion: Any,
                    print_every: int,
                    dev_dataset: Union[dataset, None],
                    train_dataset: dataset,
                    batch_size: int,
                    no_eval_batches: int) -> None:
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
        :param no_eval_batches: number of random batches to be assessed every `print_every` iterations.
        :return:
        """

        # retrieve the number of total iterations of one epoch
        no_iterations: int = len(train_iterator)

        # Get start timestamp
        start: float = time.time()

        # accumulated loss over `print_every` iterations
        acc_loss: float = 0.

        # perform a forward and backward pass on the batch
        # Each batch is a tuple (X,y)
        iter: int
        batch: Dict[str,Any]
        for iter, batch in enumerate(train_iterator, start=1):
            print('.', end='', flush=True)

            # fp and bp
            loss: float = self.train(batch,
                              optimizer,
                              lr_scheduler,
                              loss_criterion)

            acc_loss += loss

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

                # report loss and perf. metrics
                self.report(train_dataset,
                            dev_dataset,
                            batch_size,
                            acc_loss,
                            print_every,
                            start,
                            no_iterations,
                            iter,
                            no_eval_batches)
                acc_loss = 0.



    def report(self,
               train_dataset: dataset,
               dev_dataset: Optional[dataset],
               batch_size: int,
               acc_loss: float,
               print_every: int,
               start: float,
               no_iterations: int,
               iter: int,
               no_batches: int
               ) -> None:
        """
        Reports loss and performance metrics on train and, if provided, also on the dev dataset.
        :param train_dataset: training dataset
        :param dev_dataset: dev dataset (can be None) -- in this case the model won't be evaluated on dev data.
        :param batch_size: batch size
        :param acc_loss: accumulated loss over the last `print_every` iterations
        :param print_every: number of iterations needed to call this method
        :param start: training start time (to report elapsed time)
        :param no_iterations: number of batches in one epoch, i.e. number of iterations to finish one epoch.
        :param iter: current iteration
        :param no_batches: number of random batches to be evaluated
        :return:
        """

        # Evaluate and on train dataset
        self.evaluate(train_dataset, batch_size, 'Train', no_batches=no_batches, plot=False, step=iter)

        # Evaluate on test dataset if provided
        if dev_dataset:
            self.evaluate(dev_dataset, batch_size, 'Dev', no_batches=no_batches, plot=False, step=iter)

        loss_avg: float = acc_loss / print_every
        teletype.print_information('%s (%d %d%%) loss: %.4f' % (utils.time_since(start, iter / no_iterations),
                                           iter,
                                           iter / no_iterations * 100,
                                           loss_avg,
                                           ), __name__)
        # also log the metric
        self.glog.log_metrics(
            {
                'metrics': {
                    'avg_train_loss': loss_avg
                },
                'step': iter
            }
        )

    def train(self,
              batch: Dict[str, Any],
              optimizer: Any,
              lr_scheduler: Any,
              loss_criterion: Any) -> float:
        """
        Performs a forward and backwards pass on the batch
        :param batch: training batch
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
                             batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """
        Performs a forward pass using the underlying network
        :param batch: training batch
        :return: output of the network (shape [batch_size, n_classes]) and gold labels (shape [batch_size])
        """

        return self.eat(**batch), batch['y']


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

        teletype.start_subtask(f'Evaluate: "{evaluate_label}"', __name__, 'evaluate')

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

        # iterate batches
        iter: int
        batch: Dict[str,Any]
        for iter, batch in enumerate(batch_iterator, start=1):
            # get ground truth
            y: Tensor = batch['y'] # y[batch_size]

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

        # leave evaluation mode
        self.eat.train()

        teletype.finish_subtask(__name__, 'evaluate')

        return ys_gt, ys_hat

    def predict(self,
                batch: Dict[str,Any]) -> Tensor:
        """
        Retrieves the predicted labels from the X data of a batch
        :param batch: batch parameters
        :return: predicted labels (y_hat) of shape [batch_size]
        """

        # Perform a forward pass
        output: Tensor # output[batch_size, n_classes]
        output, _ = self.perform_forward_pass(batch)

        # Retrieve the index of that element with highest log probability from the softmax classification
        y_hat: Tensor = output.topk(1)[1] # y_hat[batch_size]

        return y_hat
