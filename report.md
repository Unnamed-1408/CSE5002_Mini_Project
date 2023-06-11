# CSE5002 Project

> Name : 谢岳臻	SID : GTM11913008

## Problem Statement

> Key Points : What is the problem to solve?

This project can be simplified to classify the nodes in the graph, which is mainly a multi-class classification problem in graph. Given the feature of each nodes, and the whole graph (the nodes are connected with edges), then we can predict the nodes' labels.

## Data Processing

> Key Points : How do you process the data (including topology and attributes) before feeding to a classifier?

Given the feature of each nodes $X \in \mathbb{R}^{n\times6}$ , edges are expressed with a adjlist file. We can easily change the adjlist to adjacency matrix or edges.

### Raw Features

The raw features contains six features, and each features can describe as followed

| Describe\Index | degree   | gender   | major    | second_major | dormitory  | high_school  |
| -------------- | -------- | -------- | -------- | ------------ | ---------- | ------------ |
| mean           | 1.440060 | 1.537474 | 9.067963 | 7.519351     | 208.175382 | 18515.809137 |
| std            | 0.930884 | 0.548758 | 7.566342 | 10.132974    | 85.977516  | 18476.478081 |
| min            | 1        | 0        | 0        | 0            | 0          | 0            |
| max            | 6        | 2        | 44       | 44           | 290        | 61426        |

The minimize value 0 is the **absent value**, we need to full them. I choose to use the `SimpleImputer` in  `sklearn`. Fit the learner in the training set, and use it both in training set and testing set.

```python
sim = SimpleImputer(missing_values=0, strategy='mean')
sim.fit(data[:4000])
data = sim.transform(data)
```

It is easy to see that each feature is different, with the 5th feature having an extremely large mean and variance. If the unprocessed data is directly to the classifier, the 5th feature will play a large role in the discourse. Therefore, each feature needs to be **normalized** before input to the classifier.

In this project, I use the *StandardScaler* in *sklearn*.
$$
z = \frac{(x - u)}{s}
$$
However, **gender**, **major**, **second_major**, **school** should not compare with the values. So, we can use **one-hot** encoding to get feature. Since the **high school** has  too many values, I only one-hot encoding **gender**, **major**, **second_major** three features. The feature after one-hot encoding will be $X \in \mathbb{R}^{n\times94}$.

|   Feature    | number of classes |
| :----------: | :---------------: |
|    degree    |         6         |
|    gender    |         3         |
|    major     |        43         |
| second_major |        44         |
|  dormitory   |        64         |
| high_school  |       2505        |

In the experiment section, I will show the result both without one-hot encoding and with one-hot encoding.

### Edges

Transforming the adjlist into **edge** and **adjacency** matrix makes it easy to input to subsequent processing, mainly to facilitate input to the neural network.

### Labels

The distribution of the labels can be visualized in the following diagram:

Training set :

![image-20230610213901442](/home/bill/.config/Typora/typora-user-images/image-20230610213901442.png)

Testing set :

![image-20230610213920534](/home/bill/.config/Typora/typora-user-images/image-20230610213920534.png)

It is easy to see that the number of people in each category (year) is extremely unevenly distributed, so direct use of the classifier for classification may make the classifier more biased towards classes with a high number in the training set.

In addition to that, on the test set, years appearing that are not in the training set can be marked as error points, but they still have to be calculated when calculating the accuracy. The following years are not appeared in training set.

* 1900
* 1956
* 1979

Then, the years should be mapped into a sequences of number, instead of spares years data. Using `labels_uni, indices = np.unique(labels, return_inverse=True)` , we can get the unique labels and the `inv_list` to transform back to the years data. The mapping can see as followed.

| Mapping Lables | Years | Mapping Lables | Years | Mapping Lables | Years |
| -------------- | ----- | :------------: | ----- | :------------: | :---: |
| 0              | 1928  |       5        | 1977  |       10       | 1990  |
| 1              | 1947  |       6        | 1980  |       11       | 1993  |
| 2              | 1968  |       7        | 1981  |       12       | 1994  |
| 3              | 1975  |       8        | 1987  |       13       | 1995  |
| 4              | 1976  |       9        | 1989  |     14....     | ....  |

Considering the extreme imbalance of samples and the fact that there are many classes with only one sample, the number of samples is balanced using **resampling**.  This may leads to over-fitting.

![image-20230610221057193](/home/bill/.config/Typora/typora-user-images/image-20230610221057193.png)

## Model

After careful consideration, the models used can be classified into the following three categories

* Unsupervised Graph Embedding + Classifier
  * Two features : Graph Features, Raw Features
  * Combination ways :
    * Concatenate
    * Self-attention (weighted) 
* Supervised Graph Embedding + Classifier
* GNN (**Graphic Nuaral Network**)

### Unsupervised Graph Embedding + Classifier

A graph embedding determines a fixed length vector representation for each entity (usually nodes) in our graph. These embeddings are a lower dimensional representation of the graph and preserve the graph’s topology. And the lower dimensional representation can combine with the given raw features, then pass them to the classifier to predict the labels.

Depending on the way of combining, the methods can be divided into **direct concatenation** and the use of **self-attention** mechanisms.

The whole pipeline can seen as followed :

![1111](/home/bill/Downloads/1111.png)

* Graph Embedding
  * Lower dimensional representation of the graph
  * DeepWalk
    * Using short random walks to learn representations for vertices in graphs
  * Node2Vec
    * Extention of DeepWalk, combining with DFS and BFS, better than DeepWalk
* Combination
  * Fuse two features, $A, B$
  * Concatenate $[A, B]$
  * Self-attention $W \times[A, B]$
* PCA
  * Reducing the dimensionality of features
* Classifier
  * Predict the Labels
  * MLP
    * Fully connected class of feed-forward artificial neural network (ANN)
  * OneVsOne
    * Fitting one classifier per class pair
    * Change the multi-class classification to binary classification
  * RandomForest
    * Ensemble learning method => Bagging, a combination of strong learner
    * Operates by constructing a multitude of decision trees

These methods can be combined in many ways to predict labels.

### Supervised Graph Embedding + Classifier

Using neural networks to extract graph features, unlike unsupervised graph embedding, this can be trained as a layer in a neural network. In this project, the **GIN model** is used for exploration.

> Graph Isomorphism Network (GIN) generalizes the WL test and hence achieves maximum discriminative power among GNNs.

After the layer of GIN, we can simply use the MLP to predict the labels and compare with the unsupervised graph embedding ways. The whole pipeline : 

![22222](/home/bill/Downloads/22222.png)

### GNN (Graphic Nuaral Network)

GNN has various models, I have select three classical GNN to test, **GAT**, **GCN** and **GraphSAGE**.

![gnn-architectures-review](https://theaisummer.com/static/4ce0369b5414edb9f0c0859eb06ca93b/5a190/gnn-architectures-review.png)

**GCN**

Graph Convolutional Networks (GCN) is the most cited paper in the GNN literature and the most commonly used architecture in real-life applications.

GCN is the first to propose a **convolutional approach** to fusion of graph structure features, providing a new perspective. GCN is computationally well understood, essentially the same as the CNN convolution process, a weighted summation process, that is, the neighbor points through the degree matrix and its adjacency matrix, calculate the weight of each edge, and then weighted summation.

**GAT**

GCN does convolution, and the weights on the edges are fixed every time when need to do fusion. Add an attention mechanisms to the fusion and let the model learn by itself, this is what GAT doing.

It can be said that, **GAT** is the extension of **GCN**.

**GraphSAGE**

GCN does convolutional fusion based on the full graph, and the gradient is updated based on the full graph. If the graph is large and there are more neighboring nodes at each point, the efficiency of such fusion is inevitably very low.

> **Transductive Learning VS. Inductive Learning**
>
> Transductive learning : means that the data to be predicted **can be seen** by the model during training. To explain further, it means that before training, the structure of the graph is already fixed, and the structure of the points or edge relationships you want to predict should already be in this graph.
>
> Inductive learning : means that the data to be predicted is **not seen** by the model during training, which is how we normally do algorithmic models, and the data is separated during training and prediction, which means that the graph structure can be not fixed as mentioned above, and new nodes are added.

GraphSAGE is a model of inductive learning. GraphSAGE proposes to randomly pick subgraphs to sample and update node embedding through subgraphs, so that the structure of the picked subgraphs itself is changed, and thus the model learns a way of sampling and aggregating parameters, which effectively solves the unseen nodes This effectively solves the problem of unseen nodes and avoids the dilemma of updating the node embedding of the whole graph together during training, which effectively increases the scalability.

It can be said that, **GAT** is the extension of **GCN**.

Therefore, it can be expected that the performance of GAT and GraphSAGE is better than that of GCN.

### Final Model

I choose the GraphSAGE, because

* In the comparison experiment, it has the best performance
* Simple and effective
* My roommate has already helped me to realize it, and it is very simple to start experimenting...

## Evaluate Metric

### Accuracy

Accuracy can be used to measure the accuracy of a model, a very general but very common indicator of the model's performance
$$
ACC = \frac{\sum\mathbb{I}(y = \hat{y})}{|y|}
$$

### Cross Entropy Loss

It is mainly used to measure the difference between two probability distributions,  and is a **loss function** commonly used in classification problems.
$$
Loss = -\hat{y}\text{log(softmax($y$))}
$$

### Negative Log-Likelihood Loss

Minimize the difference between the model prediction results and the true label to make the model's prediction results closer to the true results. Similar to the Cross Entropy Loss
$$
\text{CrossEntropyLoss}(x, y) = \text{NLL\_Loss}(\text{log Softmax($x$), y})
$$

## Experiments Details

### Environment Setup

All the experiments are based on *python* , mainly uses *pytorch*, *sklearn* and *pytorch-geometric*. The experiments are performed both on my personal computer with AMD Ryzen 5 3600X (12) @ 3.800GHz and NVIDIA GeForce RTX 2060 SUPER, and a server with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz and 6 TITAN V.

All the requirement libraries are in README.md in code folds, you can also check the requirement on [github repo][https://github.com/Unnamed-1408/CSE5002_Mini_Project.git].

### DeepWalk

Given the following parameters

```python
number_walks = 8
representation_size = 128
walk_length = 40
window_size = 10
```

### Node2Vec

Given the following parameters

```python
dimensions=128
walk_length=30
num_walks=200
window=10
```

### Self-Attention

The Self-Attention module takes an input tensor `x` with shape `(batch_size, seq_length, input_dim)` and returns a weighted representation of the input sequence with the same form.

The attention mechanism is implemented using **dot-product attention**, where the query, key, and value vectors are learned through linear transformations of the input sequence.

The attention scores are then calculated as the dot product of the queries and keys, and the attention is applied by **multiplying the values** by the attention scores. The result is a weighted representation of the input sequence that considers each element’s importance.

![](/home/bill/Downloads/3333.png)

Input_dim is set to 64.

### MLP

MLP with three layers, the dimensions are 128, 64, 32 respectively.

![4444](/home/bill/Downloads/4444.png)

The final output of the 32-dimensional vector represents the likelihood of the 32 classes.

MLP uses Cross Entropy Loss to train.

### OneVsOne

> PCA reduces to 128 dimensions.

Each classifier is `SVC` , SVC classification is a nonlinear classifier that separates different classes of data by constructing a nonlinear decision boundary. And can use C, to prevent overfitting.

Using `CalibratedClassifierCV `  to cross-validation the training process

```python
clf = SVC(C=1)
clf = CalibratedClassifierCV(clf, method='sigmoid')
clf = OneVsOneClassifier(clf)
clf.fit(X_train, Y_train)
```

### RandomForest

Also use cross-validation to train the model

```python
acc = cross_val_score(estimator=RandomForestClassifier(n_estimators=100), X=X_train, y=Y_train, cv=10)
print("average accuracy :", np.mean(acc))
print("average std :", np.std(acc))
```

### GIN

Using GIN + MLP to predict, and use `CrossEntropyLoss` totrain the whole model

```python
self.gnn = GIN(in_channels, hidden_channels, num_layers, dropout=dropout, jk='cat')
self.classifier = MLP([hidden_channels, hidden_channels, out_channels], norm="batch_norm", dropout=dropout)
```

Using Adam optimizer, learning rate set to `0.01`, weight_decay set to `5e-4`.

### GCN, GAT, GraphSAGE

All using Adam optimizer, learning rate set to `0.01`, weight_decay set to `5e-4`. Using `Negative Log-Likelihood Loss` to train the whole model

**GCN**

Using two layers Graph Convolution Layers.

```python
self.conv1 = GraphConvolution(nfeat, nhid)
self.conv2 = GraphConvolution(nhid, nclass)
```

**GAT**

Using two layers GAT Convolution Layers.

```python
self.gat1 = GATConv(feature, hidden, heads=heads)
self.gat2 = GATConv(hidden*heads, classes)
```

**GraphSAGE**

Using exist code.

## Experiment Result

> Due to the limitation of time, all the experiment is test only once, maybe random factors interfering.

### Unsupervised Graph Embedding + Classifier

Final Result shows in the table(**fill the absent values, Without one-hot encoding**):

|                     | DeepWalk + MLP     | DeepWalk + MLP + Self-attention | DeepWalk + OVO      | DeepWalk + RandomForest |
| ------------------- | ------------------ | :-----------------------------: | ------------------- | ----------------------- |
| Training Acc (Best) | 0.9961987510181917 |                                 | 0.9913114308987239  | 0.9723975550452796      |
| Testing Acc         | 0.7773497688751926 |                                 | 0.8020030816640986  | 0.7372881355932204      |
| F1-Score macro      | 0.3354470491354979 |                                 | 0.35432221301417494 | /                       |
| F1-Score micro      | 0.7773497688751926 |                                 | 0.8020030816640985  | /                       |
| F1-Score weighted   | 0.7759898313239914 |                                 | 0.796546634589864   | /                       |

|                     | Word2vec + MLP     | Word2vec + MLP + Self-attention | Word2vec + OVO      | Word2vec + RandomForest |
| :-----------------: | ------------------ | :-----------------------------: | ------------------- | ----------------------- |
| Training Acc (Best) | 0.9971038102995746 |                                 | 0.9918997194316228  | 0.9771034988498307      |
|     Testing Acc     | 0.8020030816640986 |                                 | 0.8212634822804314  | 0.7604006163328197      |
|   F1-Score macro    | 0.3593160438414843 |                                 | 0.36734714062907653 | /                       |
|   F1-Score micro    | 0.8020030816640986 |                                 | 0.8212634822804314  | /                       |
|  F1-Score weighted  | 0.7932764320078353 |                                 | 0.8182858815951268  | /                       |

**Without filling the absent values**(only test a few model):

|                     | DeepWalk + MLP | DeepWalk + OVO | DeepWalk + RandomForest |
| :-----------------: | :------------: | :------------: | :---------------------: |
| Training Acc (Best) |                |                |                         |
|     Testing Acc     |                |                |                         |
|   F1-Score macro    |                |                |                         |
|   F1-Score micro    |                |                |                         |
|  F1-Score weighted  |                |                |                         |

**With the one-hot encoding**(only test a few model):

|                     | DeepWalk + MLP | DeepWalk + OVO | DeepWalk + RandomForest |
| :-----------------: | :------------: | :------------: | :---------------------: |
| Training Acc (Best) |                |                |                         |
|     Testing Acc     |                |                |                         |
|   F1-Score macro    |                |                |                         |
|   F1-Score micro    |                |                |                         |
|  F1-Score weighted  |                |                |                         |

> Why need to use PCA to reduce the dimension?

PCA **cannot** improve the accuracy if using the raw feature, which only has 6 features. The correlation matrix also shows that each feature is already conditional independent.

![image-20230611211757948](/home/bill/.config/Typora/typora-user-images/image-20230611211757948.png)

However, the graph embedding will generate 128 dimensions feature. We cannot determinate whether the features are still independent, and uses the 128 dimensions data to fit the learner will cost more times. Instead of this, using PCA can accelerate the converge speed. (It may reduce some of accuracy).



### Unsupervised Graph Embedding + Classifier

|                     | GIN + MLP |
| :-----------------: | :-------: |
| Training Acc (Best) |           |
|     Testing Acc     |           |
|   F1-Score macro    |           |
|   F1-Score micro    |           |
|  F1-Score weighted  |           |

### GNN (Graphic Nuaral Network)

|                     | GCN  | GAT  | GraphSAGE |
| :-----------------: | :--: | :--: | :-------: |
| Training Acc (Best) |      |      |    99%    |
|     Testing Acc     |      |      |    83%    |
|   F1-Score macro    |      |      |     /     |
|   F1-Score micro    |      |      |     /     |
|  F1-Score weighted  |      |      |     /     |

## Limitation and future work

