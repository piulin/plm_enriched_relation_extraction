# Enriched Attention on PLM for Low-Resource Relation Extraction

This code is part of the Master's thesis in Computational Linguistics "Exploring Linguistically Enriched Transformers 
for Low-Resource Relation Extraction".

This method aims to explore the incorporation of external linguistic knowledge from dependency parses onto a PLM  via 
enriched attention. Dependency parses constitute a rich source of linguistic information and has been used 
in many works before. A popular feature directly derived from dependency parses, The Shortest Dependency 
Path between two entities, helps in masking irrelevant words influencing the relation of entities in a sentence.
The main idea is to include the information of dependency parses similarly to the work by 
[Adel and Strötgen (2021)](https://arxiv.org/pdf/2104.10899.pdf),
but replacing the underlying LSTM recurrent neural network by a stack of transformers. 
This research direction would also allow to study the implications in terms of performance of a fine-tuned
enriched-attention PLM (e.g. RoBERTa) compared to a more traditional sequential model. 
Furthermore, it would be of interest to compare how the enriched attention mechanism impacts the performance 
depending on the underlying attention-based network.



## Installation

The provided code is written in python using a bunch of libraries, including Pytorch and Transformers. 
Using your favourite package manager, create a new environment (only `conda` was tested), and install the dependencies
included in `requirements.txt`.

If you are using conda, then type in your favourite shell, replacing `<env>` with a valid environment name:
```bash
$ conda create --name <env> --file requirements.txt
```

## Usage

To run the code, please, run the `main.py` file inside your environment e.g. typing:
```bash
$ python main.py --help
```

To set up the batch size and the computing device (GPU supported) parameters, follow the syntax:
```bash
usage: main.py [-h] [--batch-size BATCH_SIZE] [--cuda-device gpu_id]
               {train} ...

```

Currenty, the code can only train models given a dataset. The general training process 
can be tuned by the following parameters:

```bash
usage: main.py train [-h] [--experiment-label EXECUTION_LABEL]
                     [--run-label RUN_LABEL] [--disable-mlflow] --tacred
                     TACRED [--learning-rate PLM PTL] [--epochs NO_EPHOCS]
                     [--print-every no_iterations]
                     [--plm-model-path PLM_MODEL_PATH]
                     [--figure-folder FIGURE_FOLDER] [--seed SEED]
                     {Standard,ESS,Enriched_Attention} ...
```
To learn more, please check the help by typing `python main.py train --help`

The training process allows to set up and learn three different classification models for relation
extraction. On the one hand, schemas `Standard` and `ESS` are comprehensively described in the work by 
[Soares et al. (2019)](https://arxiv.org/pdf/1906.03158.pdf). Please, read section 3.2 to learn more. 
On the other hand, the schema `Enriched_Attention` is an adaptation of the method proposed by 
[Adel and Strötgen (2021)](https://arxiv.org/pdf/2104.10899.pdf), replacing the underlying LSTM by a 
PLM.

For the `Enriched_Attention` schema, further parameters must be provided:
```bash
usage: main.py train Enriched_Attention [-h]
                                        [--dependency-distance-size DEPENDENCY_DISTANCE_SIZE]
                                        [--position-embedding-size POSITION_EMBEDDING_SIZE]
                                        [--attention-size ATTENTION_SIZE]
```

Type `python main.py train Enriched_Attention --help` to learn more.

## Datasets

As of today, the code only accepts TACRED-like datasets (JSON files). Parameter `--tacred` allows you to
indicate the folder where the tacred splits are located. This folder must contain files `train.json`, `dev.json`,
and `test.json` which correspond to the train, development and test splits, respectively.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
TBD