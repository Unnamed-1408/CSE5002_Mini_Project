import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score

import data_processing
from loguru import logger
from model import node_embedding, GAT, MLP_Model, Predict, GCN, GIN_Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn.functional as F
import networkx as nx

def MLPmain():
    # data load
    logger.info("Reading Data")
    attr = data_processing.read_attr('data/attr.csv')
    edge = data_processing.read_edgelist('data/adjlist.csv')
    label_train = data_processing.read_label('data/label_train.csv')
    label_test = data_processing.read_label('data/label_test.csv')

    # define embedding models
    logger.info("Embedding Nodes")
    embedding_model = node_embedding('deepwalk')
    index, feature_vectors = embedding_model.process_embedding(edge)

    # concentrate to features
    logger.info("Feature Extraction")
    feature1 = np.zeros([attr.shape[0], feature_vectors.shape[1]])
    feature1[index] = feature_vectors
    feature2 = attr[:, 1:]
    feature = np.hstack([feature1, feature2])

    # using PCA to reduce feature dimension
    logger.info("PCA reduce to 64 dimensions")
    PCA_learner = PCA(n_components=64)
    feature_dim64 = PCA_learner.fit_transform(feature)

    # Mapping the y lables
    logger.info("Mapping Labels")
    labels = np.vstack([label_train, label_test])
    labels_uni, indices = np.unique(labels[:, 1], return_inverse=True)

    # regulartion X
    X_train = feature_dim64[:4000]
    X_test = feature_dim64[4000:]

    sc = StandardScaler()
    sc.fit(feature_dim64)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    # y labels
    labels_train = indices[:4000]
    labels_test = indices[4000:]

    # # split validation set
    # X_train, X_valid, labels_train, labels_valid = train_test_split(X_train, labels_train, test_size=0.2, random_state=42, stratify=labels_train)
    # resample
    logger.info("Resampling")
    ros = RandomOverSampler(random_state=0)
    X_train, labels_train = ros.fit_resample(X_train, labels_train)

    # train the MLP_Model to predict
    logger.info("MLP_Model Prediction")
    model = Predict('mlp', 64, 32)
    model.train(X_train, X_test, labels_train, labels_test)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def GNNmain():
    # data load
    logger.info("Reading Data")
    attr = data_processing.read_attr('data/attr.csv')
    edge = data_processing.read_edgelist('data/adjlist.csv')
    label_train = data_processing.read_label('data/label_train.csv')
    label_test = data_processing.read_label('data/label_test.csv')

    # Mapping the y lables
    logger.info("Mapping Labels")
    labels = np.vstack([label_train, label_test])
    labels_uni, indices = np.unique(labels[:, 1], return_inverse=True)

    labels_train = indices[:4000]
    labels_test = indices[4000:]
    Y_train = torch.from_numpy(labels_train)
    Y_test = torch.from_numpy(labels_test)

    # X
    X = attr[:, 1:]
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X = torch.from_numpy(X)
    X = X.type(torch.float32)
    X_train = X[:4000]
    X_test = X[4000:]
    node_idx = np.arange(0, X.shape[0])
    node_idx = torch.from_numpy(node_idx)

    edge = torch.from_numpy(edge)

    # GAT \ GCN + MLP
    # model = GAT(6, 64, 32, heads=8).to('cuda')
    model = GCN(6, 128, 32, 0.1).to('cuda')
    # model = GIN_Model(6, 32, 64, 3, 0.2).to('cuda')

    logger.info("Model Training")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    best_acc = -1
    best_out = None
    test_acc_best = -1
    for epoch in range(10000):
        optimizer.zero_grad()
        out = model(X.to('cuda'), edge.T.to('cuda'))
        loss = F.nll_loss(out[:4000], Y_train.to('cuda'))
        # loss = torch.nn.CrossEntropyLoss()(out[:4000], Y_train.to('cuda'))
        acc_train = accuracy(out[:4000], Y_train.to('cuda'))
        acc_test = accuracy(out[4000:], Y_test.to('cuda'))
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch + 1}, loss:{loss.item()}, train-acc:{acc_train}, test-acc:{acc_test}")
        if acc_train > best_acc:
            test_acc_best = acc_test
            best_out = out[4000:].max(1)[1].type_as(labels_test).cpu().data
    # F1-score
    print(f'Best Acc : {test_acc_best}')
    print('F1-Score macro: ', f1_score(labels_test, best_out, average='macro'))
    print('F1-Score micro: ', f1_score(labels_test, best_out, average='micro'))
    print('F1-Score weighted: ', f1_score(labels_test, best_out, average='weighted'))

if __name__ == "__main__":
    MLPmain()