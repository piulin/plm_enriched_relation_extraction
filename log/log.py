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
log class: it is responsible of keeping track of the training hyperparameters, as well as the execution logs.
"""

import mlflow
from datetime import datetime

class log (object) :

    def __init__(self, args):
        """
        Initializes mlflow
        :param args: command-line arguments
        """

        # Name the execution
        mlflow.set_experiment( args['experiment_label'] )

        # Label the current time
        mlflow.start_run(run_name=datetime.today().strftime('%Y-%m-%d'))

        # Log command-line arguments
        for k, v in args.items():
            self.log_param(k, v)


    def log_param(self, k, v):
        """
        Adds a parameter to the log
        :param k: key
        :param v: value
        :return:
        """
        mlflow.log_param(k,v)

    def log_metrics(self, dictionary, step):
        """
        Logs a metric
        :param step: identifier
        :param dictionary: containing pairs key value
        :return:
        """
        mlflow.log_metrics(metrics=dictionary, step=step)



