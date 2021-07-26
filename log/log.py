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
from typing import Any, Dict

"""
log class: it is responsible of keeping track of the training hyperparameters, as well as the execution logs.
"""

import mlflow

class log (object) :

    def __init__(self,
                 args: dict):
        """
        Initializes mlflow
        :param args: parameters to be logged
        """

        self.enabled: bool = not args ['disable_mlflow']

        # if not disabled by command-line, the init a new run
        if self.enabled:

            # Name the execution
            mlflow.set_experiment( args['experiment_label'] )

            # Label the current time
            mlflow.start_run(run_name=args['run_label'])

            # Log command-line arguments
            for k, v in args.items():
                self.log_param(k, v)

        else:
            print('MLFlow is disabled.')

    def log_param(self,
                  k: str,
                  v: Any) -> None:
        """
        Adds a parameter to the log
        :param k: key
        :param v: value
        :return:
        """
        if self.enabled:
            mlflow.log_param(k,v)

    def log_metrics(self,
                    args: Dict[str, Any]) -> None:
        """
        Logs a metric
        :param step: identifier
        :param dictionary: containing pairs key value
        :return:
        """
        if self.enabled:
            mlflow.log_metrics( ** args )

    def log_artifact(self,
                     path_to_artifact: str) -> None:
        """
        Logs files into mlflow
        :param path_to_artifact: path to the file to be log
        :return:
        """
        if self.enabled:
            mlflow.log_artifact(path_to_artifact)



