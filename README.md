# CSE5002_Mini_Project

A hands-on course mini-project of CSE5002 Intelligent Data Analysis, SUSTech 2023. The project aims to have students evaluate some classification models using the given dataset. This repo mainly focused on Graph Embedding + MLP and GNN models, all the experiment details can be found in report fold.

The model implementation uses [PyTorch](https://pytorch.org/) and [Pytorch_geometric](https://pyg.org/). You can find all the models in my [github repo ][https://github.com/Unnamed-1408/CSE5002_Mini_Project].

## Dependency

The required python packages are listed below:

* Pytorch with cuda support
* [pytorch_geometric][https://github.com/pyg-team/pytorch_geometric]
* numpy
* scikit-learn
* loguru
* [imbalanced-learn](https://imbalanced-learn.org/)
* [NetworkX](https://networkx.org/)

Or you can directly reproduce the environment on my local machine by

```bash
conda env create -f environment.yml
```

## Content 

* data
  * dataset : contains adjlist, attr, labels
* models
  * third-party models : adapt from [deepwalk][https://github.com/phanein/deepwalk]
  * third-party models (fom my roommate) : GraphSAGE
* output
  * need to create manually
  * save the intermediate files of the deepwalk algorithm run
* report
  * report markdown file and pdf
* data_processing,py
  * preprocessing the dataset
* main.py
  * the training entrance
* model.py
  * different models (MLP/GNN...)

## How To Run

> Under the linux platform

Steps 1: create `output` fold

```bash
mkdir output
```

Step 2: run the following code

```bash
conda activate graph
python main.py
```

Or you can add more parameters to change the model to train

```bash
python main.py -h

usage: main.py [-h] [-m MODEL_NAME] [-e EMBEDDING]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        Running Model Name, default = onevsone
  -e EMBEDDING, --embedding EMBEDDING
                        define the unsupervised graph embedding method, default = node2vec
```

You can choose the following models : `mlp`, `onevsone`, `randomforest`, `gin`, `gcn`, `gat`

The first three models can define the embedding ways : `deepwalk`, `node2vec`

### Example

```bash
> python main.py -m mlp -e deepwalk                                                                               

2023-06-13 15:46:02.260 | INFO     | __main__:MLPmain:16 - Reading Data
2023-06-13 15:46:03.422 | INFO     | __main__:MLPmain:23 - Embedding Nodes
Number of nodes: 5270
Number of walks: 42160
Data size (walks*length): 1686400
Data size 1686400 is larger than limit (max-memory-data-size: 0).  Dumping walks to disk.
Walking...
Counting vertex frequency...
Training...
2023-06-13 15:46:04 WARNING word2vec.py: 1545 Both hierarchical softmax and negative sampling are activated. This is probably a mistake. You should set either 'hs=0' or 'negative=0' to disable one of them.
2023-06-13 15:46:39.029 | INFO     | __main__:MLPmain:28 - Feature Extraction
2023-06-13 15:46:39.030 | INFO     | __main__:MLPmain:35 - PCA reduce to 64 dimensions
2023-06-13 15:46:39.031 | INFO     | __main__:MLPmain:40 - Mapping Labels
2023-06-13 15:46:39.036 | INFO     | __main__:MLPmain:60 - Resampling
2023-06-13 15:46:39.046 | INFO     | __main__:MLPmain:65 - MLP_Model Prediction
epoch:1, loss:3.4848451614379883, acc_train:0.03860077835098199, acc_test:0.11016949152542373
epoch:2, loss:3.1372647285461426, acc_train:0.3472259933025613, acc_test:0.02157164869029276
epoch:3, loss:2.7814383506774902, acc_train:0.4872839170965698, acc_test:0.012326656394453005
epoch:4, loss:2.354548931121826, acc_train:0.5095483754185899, acc_test:0.023112480739599383
......
epoch:9997, loss:0.019466444849967957, acc_train:0.9965607747307449, acc_test:0.7858243451463791
epoch:9998, loss:0.021434931084513664, acc_train:0.9957462213775002, acc_test:0.7896764252696457
epoch:9999, loss:0.02094126120209694, acc_train:0.99416236763508, acc_test:0.7750385208012327
epoch:10000, loss:0.02181904949247837, acc_train:0.9958367273056385, acc_test:0.7935285053929122
Best Acc : 0.7935285053929122
F1-Score macro:  0.3299152425861241
F1-Score micro:  0.7935285053929121
F1-Score weighted:  0.7920860906568141
```

### GraphSAGE

No need to install the dependency again, directly run

```bash
conda activate graph
sh script.sh
```

